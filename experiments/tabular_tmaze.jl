module TabularTMazeExperiment

import Flux: Descent
import Random

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays


const TTMU = Curiosity.TabularTMazeUtils

default_args() =
    Dict(
        # Behaviour Items
        "behaviour_eta" => 0.2,
        "behaviour_gamma" => 0.9,
        "behaviour_learner" => "Q",
        "behaviour_update" => "TabularRoundRobin",
        "behaviour_trace" => "accumulating",
        "behaviour_opt" => "Descent",
        "behaviour_lambda" => 0.9,
        "exploration_param" => 0.2,
        "exploration_strategy" => "epsilon_greedy",

        # Demon Attributes
        "demon_alpha_init" => 0.1,
        "demon_eta" => 0.1,
        "demon_discounts" => 0.9,
        "demon_learner" => "Q",
        "demon_update" => "SARSA",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Descent",
        "demon_lambda" => 0.9,

        # Environment Config
        "constant_target"=> 1.0,
        "cumulant_schedule" => "DrifterDistractor",
        "distractor" => (1.0, 1.0),
        "drifter" => (1.0, sqrt(0.01)),
        "exploring_starts"=>true,

        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        "logger_keys" => [LoggerKey.TTMAZE_ERROR],
        "save_dir" => "TabularTMazeExperiment",
        "seed" => 1,
        "steps" => 5000,
        "use_external_reward" => true,
    )


function construct_agent(parsed)

    feature_size = 21
    action_space = 4
    observation_size = 5
    demon_alpha_init = parsed["demon_alpha_init"]
    demon_learner = parsed["demon_learner"]
    demon_lu = parsed["demon_update"]

    behaviour_learner = parsed["behaviour_learner"]
    behaviour_lu = parsed["behaviour_update"]
    behaviour_discount = parsed["behaviour_gamma"]

    intrinsic_reward_type = parsed["intrinsic_reward"]
    behaviour_trace = parsed["behaviour_trace"]
    use_external_reward = parsed["use_external_reward"]

    #Create state constructor
    function state_constructor(observation, feature_size)
        s = spzeros(feature_size)
        s[convert(Int64,observation[1])] = 1
        return s
    end

    demons = get_horde(parsed, feature_size, action_space, (obs) -> state_constructor(obs, feature_size))

    exploration_strategy = if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["exploration_param"])
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end

    demon_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 demons,
                                                 "demon")
    behaviour_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 demons,
                                                 "behaviour")

    behaviour_gvf = TTMU.make_behaviour_gvf(behaviour_discount, (obs) -> state_constructor(obs, feature_size), behaviour_learner, exploration_strategy)
    behaviour_demons = if behaviour_learner isa GPI
        SF_horde = TTMU.make_SF_horde(behaviour_discount, feature_size, action_space)
        num_SFs = 4

        pred_horde = Horde([behaviour_gvf])

        Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, (obs) -> state_constructor(obs, feature_size))
    elseif behaviour_learner isa QLearner
        Horde([behaviour_gvf])
    end


    Agent(demons,
          feature_size,
          behaviour_learner,
          behaviour_demons,
          behaviour_discount,
          demon_learner,
          observation_size,
          action_space,
          intrinsic_reward_type,
          (obs) -> state_constructor(obs, feature_size),
          use_external_reward,
          exploration_strategy)
end

function get_horde(parsed, feature_size, action_space, state_constructor)

    discount = parsed["demon_discounts"]
    pseudoterm = TTMU.pseudoterm
    num_actions = TTMU.NUM_ACTIONS
    num_demons = TTMU.NUM_DEMONS


    #TODO: Sort out the if-else block so that demon_policy_type and horde_type is not blocking eachother.
    horde = if parsed["demon_policy_type"] == "greedy_to_cumulant" && parsed["horde_type"] == "regular"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.FunctionalPolicy((;kwargs...) -> TTMU.demon_target_policy(i;kwargs...))) for i in 1:num_demons])
    elseif parsed["demon_policy_type"] == "random" && parsed["horde_type"] == "regular"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions))) for i in 1:num_demons])
    else
        throw(ArgumentError("Not a valid policy type for demons"))
    end

    if parsed["demon_learner"] == "SR"
        num_SFs = 4
        SF_horde = TTMU.make_SF_horde(discount, feature_size, action_space)

        horde = Curiosity.GVFSRHordes.SRHorde(horde, SF_horde, num_SFs, state_constructor)
    end

    return horde
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    Random.seed!(parsed["seed"])

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)

    agent = construct_agent(parsed)

    goal_visitations = zeros(4)

    logger_init_dict = Dict(
        LoggerInitKey.TOTAL_STEPS => num_steps,
        LoggerInitKey.INTERVAL => 50,
        LoggerInitKey.ENV => "tabular_tmaze"
    )

    Curiosity.experiment_wrapper(parsed, logger_init_dict, working) do parsed, logger
        eps = 1
        max_num_steps = num_steps
        steps = Int[]

        while sum(steps) < max_num_steps
            cur_step = 0
            max_episode_steps = min(max_num_steps - sum(steps), 1000)
            tr, stp =
                run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
                    #This is a callback for every timestep where logger can go
                    # agent is accesible in this scope

                    if t == true
                        goals = s_next[2:end]
                        f = findfirst(!iszero, goals)
                        goal_visitations[f] += 1
                    end

                    logger_step!(logger, env, agent, s, a, s_next, r, t)
                    cur_step+=1
                end
                logger_episode_end!(logger)

            push!(steps, stp)
            eps += 1
        end
        println(goal_visitations)
    end

end

end
