module TabularTMazeExperiment

import Flux: Descent, ADAM
import Random

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays
using ProgressMeter

const TTMU = Curiosity.TabularTMazeUtils

default_args() =
    Dict(
        # Behaviour Items
        "behaviour_eta" => 0.55,
        "behaviour_gamma" => 0.9,
        "behaviour_learner" => "Q",
        "behaviour_update" => "TabularRoundRobin",
        "behaviour_trace" => "ReplacingTraces",
        "behaviour_opt" => "Descent",
        "behaviour_lambda" => 0.9,
        "exploration_param" => 0.3,
        "exploration_strategy" => "epsilon_greedy",

        # Demon Attributes
        "demon_alpha_init" => 1.0,
        "demon_eta" => 0.4,
        "demon_discounts" => 0.9,
        "demon_learner" => "LSTD",
        "demon_update" => "TB",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Descent",
        "demon_lambda" => 0.9,
        "demon_trace"=> "AccumulatingTraces",

        # Environment Config
        "constant_target"=> 1.0,
        "cumulant_schedule" => "DrifterDistractor",
        "distractor" => (1.0, 1.0),
        "drifter" => (sqrt(0.01), 1.0),
        "exploring_starts"=>true,

        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        "logger_keys" => [LoggerKey.TTMAZE_ERROR, LoggerKey.TTMAZE_UNIFORM_ERROR, LoggerKey.GOAL_VISITATION],
        "save_dir" => "TabularTMazeExperiment",
        "seed" => 1,
        "steps" => 1000,
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
    state_constructor_func = (obs) -> state_constructor(obs, feature_size)

    function compressed_state_constructor(obs)
        s = spzeros(4)
        if obs[1] < 5
            s[1] = 1
        elseif obs[1] < 8
            s[2] = 1
        elseif obs[1] < 15
            s[3] = 1
        elseif obs[1] < 17
            s[2] = 1
        else
            s[4] = 1
        end
        return s
    end

    behaviour_feature_projector = ActionValueFeatureProjector(state_constructor_func, feature_size)
    # behaviour_feature_projector = ActionValueFeatureProjector(compressed_state_constructor, 4)
    demon_feature_projector = behaviour_feature_projector


    demons = get_horde(parsed, length(demon_feature_projector), action_space, demon_feature_projector)

    exploration_strategy = if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["exploration_param"])
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end

    demon_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 demons,
                                                 "demon",
                                                 demon_feature_projector)



    # TODO: Behaviour horde needs access to the behaviour learner to condition the behaviour policy
    # BUT behaviour learner needs access the horde to know things like how many demons there are.
    behaviour_num_tasks = 1
    num_SFs = 4
    num_demons = if parsed["behaviour_learner"] ∈ ["GPI"]
        num_SFs * feature_size * action_space + 1
    elseif parsed["behaviour_learner"] ∈ ["Q"]
        1
    end

    behaviour_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 num_demons,
                                                 behaviour_num_tasks,
                                                 "behaviour",
                                                 behaviour_feature_projector)

    behaviour_gvf = TTMU.make_behaviour_gvf(behaviour_discount, state_constructor_func, behaviour_learner, exploration_strategy)
    behaviour_demons = if behaviour_learner isa GPI
        SF_horde = TTMU.make_SF_horde(behaviour_discount, feature_size, action_space, behaviour_feature_projector)

        pred_horde = Horde([behaviour_gvf])

        Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, behaviour_feature_projector)
    elseif behaviour_learner isa QLearner
        Horde([behaviour_gvf])
    else
        throw(ArgumentError("goes with which horde? " ))
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

function get_horde(parsed, feature_size, action_space, projected_feature_constructor)

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
        SF_horde = TTMU.make_SF_horde(discount, length(projected_feature_constructor), action_space, projected_feature_constructor)

        horde = Curiosity.GVFSRHordes.SRHorde(horde, SF_horde, num_SFs, projected_feature_constructor)
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

        if progress
            p = Progress(max_num_steps)
        end

        while sum(steps) < max_num_steps
            cur_step = 0
            max_episode_steps = min(max_num_steps - sum(steps), 1000)
            tr, stp =
                run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
                    #This is a callback for every timestep where logger can go
                    # agent is accesible in this scope

                    if t == true && working==true && t
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

            if progress
                ProgressMeter.update!(p, sum(steps))
            end
        end
        if working == true
            println(goal_visitations)
        end
    end

end

end
