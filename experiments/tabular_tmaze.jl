module TabularTMazeExperiment

using GVFHordes
using Curiosity
using MinimalRLCore

const TTMU = Curiosity.TabularTMazeUtils

default_args() =
    Dict(
        "lambda" => 0.9,
        "demon_alpha" => 0.1,
        "demon_alpha_init" => 1.0,
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_learner" => "TBAuto",
        "steps" => 2000,
        "seed" => 1,
        "cumulant_schedule" => "DrifterDistractor",
        "drifter" => (1.0, sqrt(0.01)),
        "distractor" => (1.0, 1.0),
        "constant_target"=> 1.0,
        "exploring_starts"=>true,
    )


function construct_agent(parsed)

    feature_size = 21
    action_space = 4
    lambda = parsed["lambda"]
    demon_alpha = parsed["demon_alpha"]
    demon_alpha_init = parsed["demon_alpha_init"]
    demon_learner = parsed["demon_learner"]

    demons = get_horde(parsed)

    if demon_learner == "TB"
        demon_learner = TB(lambda, feature_size, length(demons), action_space, demon_alpha)
    elseif demon_learner == "TBAuto"
        demon_learner = TBAuto(lambda, feature_size, length(demons), action_space, demon_alpha, demon_alpha_init)
    else
        throw(ArgumentError("Not a valid demon learner"))
    end

    behaviour_learner = TabularRoundRobin()
    agent = Agent(demons, feature_size, feature_size, action_space, demon_learner, behaviour_learner)
end

function get_horde(parsed)
    num_gvfs = 4
    num_actions = 4
    discount = 0.9
    function pseudoterm(state)
        term = false
        if state[1] == 1 || state[1] == 5 || state[1] == 17 || state[1] == 21
            term = true
        end
        return term
    end

    function demon_target_policy(gvf_i, observation, action)
        state = convert(Int,observation[1])

        policy_1 =  [1 0 0 0 0 0 3;
                     1 0 0 0 0 0 3;
                     1 4 4 4 4 4 4;
                     1 0 0 1 0 0 1;
                     1 0 0 1 0 0 1;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0]

        policy_2  = [3 0 0 0 0 0 3;
                     3 0 0 0 0 0 3;
                     3 4 4 4 4 4 4;
                     3 0 0 1 0 0 1;
                     3 0 0 1 0 0 1;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0]
        policy_3 = [3 0 0 0 0 0 1;
                     3 0 0 0 0 0 1;
                     2 2 2 2 2 2 1;
                     1 0 0 1 0 0 1;
                     1 0 0 1 0 0 1;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0]
        policy_4 = [3 0 0 0 0 0 3;
                     3 0 0 0 0 0 3;
                     2 2 2 2 2 2 3;
                     1 0 0 1 0 0 3;
                     1 0 0 1 0 0 3;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0;
                     0 0 0 1 0 0 0]
        gvfs = [policy_1,policy_2,policy_3, policy_4]
        mask = valid_state_mask()

        action_prob = if gvfs[gvf_i][mask][state] == action
            1.0
        else
            0.0
        end
        return action_prob
    end

    horde = if parsed["demon_policy_type"] == "greedy_to_cumulant"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.FunctionalPolicy((obs,a) -> demon_target_policy(i-1,obs,a))) for i in 2:5])
    elseif parsed["demon_policy_type"] == "random"
        Horde([GVF(GVFParamFuncs.FeatureCumulant(i), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions))) for i in 2:5])
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

    eps = 1
    max_num_steps = num_steps
    steps = Int[]
    goal_visitations = ones(4)
    while sum(steps) < max_num_steps
        cur_step = 0
        tr, stp =
            run_episode!(env, agent) do (s, a, s_next, r)
                #This is a callback for every timestep where logger can go
                # agent is accesible in this scope
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
