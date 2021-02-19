module TabularDrifterDistractorExperiment

using GVFHordes
using Curiosity

const TTMU = Curiosity.TabularTMazeUtils

default_args() =
    Dict(
        "lambda" => 0.9,
        "demon_alpha" => 0.5,
        "steps" => 15000,
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
    lambda = 0.9
    demon_alpha = 0.5

    demons = get_horde(parsed)
    demon_learner = TB(lambda, feature_size, length(demons), action_space, demon_alpha)
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
    horde = Horde([GVF(GVFParamFuncs.FeatureCumulant(i), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions))) for i in 2:5])
    return horde
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    seed = parsed["seed"]

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)

    agent = construct_agent(parsed)

    obs,r, is_terminal = env_start!(env)

    for i in 1:num_steps
        action = step!(agent,obs,r,is_terminal)
        obs, r, is_terminal =  env_step!(env,action)
        if is_terminal
            agent_end!(agent, obs, r, is_terminal)
            obs, r, is_terminal = env_start!(env)
        end
    end

    constant_demon = agent.demon_weights[5:8,:]
    println("Estimate for moving into constant goals: ", constant_demon[3, 3:5], " should be ~[0.225, 1.0, something]")
    println("Estimate for moving into Junction: ", constant_demon[4, 6], " should be ~0.0506")

end

end
