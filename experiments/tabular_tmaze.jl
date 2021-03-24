module TabularTMazeExperiment

import Flux: Descent
import Random

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays
using ProgressMeter



const TTMU = Curiosity.TabularTMazeUtils

default_args() =
    Dict(
        "behaviour_alpha" => 0.2,
        "behaviour_gamma" => 0.9,
        "behaviour_learner" => "RoundRobin",
        "behaviour_trace" => "accumulating",
        "constant_target"=> 1.0,
        "cumulant_schedule" => "DrifterDistractor",
        "demon_alpha_init" => 1.0,
        "demon_alpha" => 0.5,
        "demon_discounts" => 0.9,
        "demon_learner" => "SR",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_update" => "TB",
        "distractor" => (1.0, 1.0),
        "drifter" => (1.0, sqrt(0.01)),
        "exploring_starts"=>true,
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        "lambda" => 0.9,
        "logger_keys" => [LoggerKey.TTMAZE_ERROR],
        "save_dir" => "TabularTMazeExperiment",
        "seed" => 1,
        "steps" => 20000,
        "use_external_reward" => true,
    )


function construct_agent(parsed)

    feature_size = 21
    action_space = 4
    observation_size = 5
    lambda = parsed["lambda"]
    demon_alpha = parsed["demon_alpha"]
    demon_alpha_init = parsed["demon_alpha_init"]
    demon_learner = parsed["demon_learner"]
    demon_lu = parsed["demon_update"]

    behaviour_learner = parsed["behaviour_learner"]
    behaviour_alpha = parsed["behaviour_alpha"]
    intrinsic_reward_type = parsed["intrinsic_reward"]
    behaviour_gamma = parsed["behaviour_gamma"]
    behaviour_trace = parsed["behaviour_trace"]
    use_external_reward = parsed["use_external_reward"]

    #Create state constructor
    function state_constructor(observation, feature_size)
        s = spzeros(feature_size)
        s[convert(Int64,observation[1])] = 1
        return s
    end

    demons = get_horde(parsed, feature_size, action_space, (obs) -> state_constructor(obs, feature_size))
    behaviour_demons = nothing

    demon_lu = if demon_lu == "TB"
        TB(lambda=lambda, opt=Descent(demon_alpha))
    elseif demon_learner == "TBAuto"
        TB(lambda,
           Auto(demon_alpha, demon_alpha_init),
           feature_size, length(demons), action_space)
    else
        throw(ArgumentError("Not a valid demon learner"))
    end

    demon_learner = if demon_learner ∈ ["Q", "QLearner", "q"]
        LinearQLearner(demon_lu, feature_size, action_space, length(demons))
    elseif demon_learner ∈ ["SR", "SRLearner", "sr"]
        SRLearner(demon_lu,
                  feature_size,
                  length(demons),
                  action_space,
                  demons.num_tasks)
    else
        throw(ArgumentError("Not a valid demon learner"))
    end

    behaviour_lu = if behaviour_learner == "ESARSA"
        ESARSA(lambda, feature_size, 1, action_space, behaviour_alpha, behaviour_trace)
    elseif behaviour_learner == "RoundRobin"
        TabularRoundRobin()
    else
        throw(ArgumentError("Not a valid behaviour learner"))
    end

    Agent(demons,
          feature_size,
          behaviour_lu,
          behaviour_gamma,
          demon_learner,
          observation_size,
          action_space,
          intrinsic_reward_type,
          (obs) -> state_constructor(obs, feature_size),
          use_external_reward)
end

function get_horde(parsed, feature_size, action_space, state_constructor)

    discount = parsed["demon_discounts"]
    pseudoterm = TTMU.pseudoterm
    num_actions = TTMU.NUM_ACTIONS
    num_demons = TTMU.NUM_DEMONS


    #TODO: Sort out the if-else block so that demon_policy_type and horde_type is not blocking eachother.
    horde = if parsed["demon_policy_type"] == "greedy_to_cumulant" && parsed["horde_type"] == "regular"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.FunctionalPolicy((obs,a) -> TTMU.demon_target_policy(i,obs,a))) for i in 1:num_demons])
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

            if progress
                ProgressMeter.update!(p, sum(steps))
            end
        end

        # if agent.behaviour_learner isa GPI
        #     obs = zeros(5)
        #     obs[1] = 3
        #     action = 1
        #     SF = predict_SF(agent.behaviour_learner, agent,  agent.behaviour_weights, obs, action)
        #     println("GPI")
        #     @show SF
        # end

        #
        # if agent.demon_learner isa SRLearner
        #     obs = zeros(5)
        #     obs[1] = 3
        #     action = 1
        #     SF = predict_SF(agent.demon_learner, agent,  agent.demon_weights, obs, action)
        #     println("SF")
        #     @show SF
        # end

        println(goal_visitations)
    end

end

end
