module TabularTMazeExperiment

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays

const TTMU = Curiosity.TabularTMazeUtils

default_args() =
    Dict(
        "lambda" => 0.9,
        "demon_alpha" => 0.1,
        "demon_alpha_init" => 0.1,
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_learner" => "TB",
        "demon_discounts" => 0.9,
        "behaviour_learner" => "RoundRobin",
        "behaviour_alpha" => 0.2,
        "behaviour_gamma" => 0.9,
        "behaviour_trace" => "accumulating",
        "intrinsic_reward" => "weight_change",
        "use_external_reward" => true,
        "steps" => 20000,
        "seed" => 1,
        "cumulant_schedule" => "DrifterDistractor",
        "drifter" => (1.0, sqrt(0.01)),
        "distractor" => (1.0, 1.0),
        "constant_target"=> 1.0,
        "exploring_starts"=>true,
        "save_dir" => "TabularTMazeExperiment",
        "logger_keys" => [LoggerKey.GOAL_VISITATION],
    )


function construct_agent(parsed)

    feature_size = 21
    action_space = 4
    observation_size = 5
    lambda = parsed["lambda"]
    demon_alpha = parsed["demon_alpha"]
    demon_alpha_init = parsed["demon_alpha_init"]
    demon_learner = parsed["demon_learner"]
    behaviour_learner = parsed["behaviour_learner"]
    behaviour_alpha = parsed["behaviour_alpha"]
    intrinsic_reward_type = parsed["intrinsic_reward"]
    behaviour_gamma = parsed["behaviour_gamma"]
    behaviour_trace = parsed["behaviour_trace"]
    use_external_reward = parsed["use_external_reward"]


    demons = get_horde(parsed)

    if demon_learner == "TB"
        demon_learner = TB(lambda, feature_size, length(demons), action_space, demon_alpha)
    elseif demon_learner == "TBAuto"
        demon_learner = TBAuto(lambda, feature_size, length(demons), action_space, demon_alpha, demon_alpha_init)
    else
        throw(ArgumentError("Not a valid demon learner"))
    end

    if behaviour_learner == "RoundRobin"
        behaviour_learner = TabularRoundRobin()
    elseif behaviour_learner == "ESARSA"
        behaviour_learner = ESARSA(lambda, feature_size, 1, action_space, behaviour_alpha)
    else
        throw(ArgumentError("Not a valid behaviour learner"))
    end

    #Create state constructor
    function state_constructor(observation,feature_size)
        s = spzeros(feature_size)
        s[convert(Int64,observation[1])] = 1
        return s
    end

    agent = Agent(demons, feature_size, feature_size, observation_size, action_space, demon_learner, behaviour_learner, intrinsic_reward_type, (obs) -> state_constructor(obs, feature_size), behaviour_gamma, use_external_reward)
end

function get_horde(parsed)

    discount = parsed["demon_discounts"]
    pseudoterm = TTMU.pseudoterm
    num_actions = TTMU.NUM_ACTIONS
    num_demons = TTMU.NUM_DEMONS

    horde = if parsed["demon_policy_type"] == "greedy_to_cumulant"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.FunctionalPolicy((obs,a) -> TTMU.demon_target_policy(i,obs,a))) for i in 1:num_demons])
    elseif parsed["demon_policy_type"] == "random"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions))) for i in 1:num_demons])
    else
        throw(ArgumentError("Not a valid policy type for demons"))
    end
    return horde
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    seed = parsed["seed"]

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)

    agent = construct_agent(parsed)

    goal_visitations = zeros(4)

    Curiosity.experiment_wrapper(parsed, working) do parsed, logger
        eps = 1
        max_num_steps = num_steps
        steps = Int[]

        while sum(steps) < max_num_steps
            cur_step = 0
            tr, stp =
                run_episode!(env, agent) do (s, a, s_next, r, t)
                    #This is a callback for every timestep where logger can go
                    # agent is accesible in this scope

                    logger_step!(logger, env, agent, s, a, s_next, r, t)

                    cur_step+=1
                    C,_,_ = get(agent.demons, s, a, s_next)
                    if sum(C) != 0
                        gvf_i = findfirst(!iszero,C)
                        goal_visitations[gvf_i] += 1
                    end
                end
            push!(steps, stp)
            eps += 1
        end
    end

    println("Took place over ", eps, " episodes")
    println("Goal Visitation Frequency")
    per = [goal_visitations[i] / sum(goal_visitations) for i in 1:4]
    println(goal_visitations)

    constant_demon = agent.demon_weights[5:8,:]
    println("Estimate for moving into constant goals: ", constant_demon[3, 3:5], " should be ~[0.9, 1.0, 0]")
    println("Estimate for moving into Junction: ", constant_demon[4, 6], " should be ~0.81")

    distractor_demon = agent.demon_weights[1:4,:]
    println("Estimate for moving into distractor goals: ", distractor_demon[1, 1:3], " should be ~[0, 1.0 , 0.9]")
    println("Estimate for moving into Junction: ", distractor_demon[4, 6], " should be ~0.81")
end

end
