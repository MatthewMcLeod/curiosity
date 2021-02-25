


module TabularTMazeExperiment

using GVFHordes
using Curiosity
using MinimalRLCore

const MCU = Curiosity.MountainCarUtils

default_args() =
    Dict(
        "steps" => 20000,
        "seed" => 1,

        "numtilings" => 16,
        "numtiles" => 16,
        
        "behaviour_learner" => "QLearning",
        "behaviour_rew" => "env",
        "behaviour_alpha" => 0.1/16,
        "behaviour_gamma" => 0.99,

        "lambda" => 0.9,
        "demon_alpha" => 0.1,
        "demon_alpha_init" => 1.0,
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_learner" => "TB",
        
        "cumulant_schedule" => "DrifterDistractor",
        "drifter" => (1.0, sqrt(0.01)),
        "distractor" => (1.0, 1.0),
        "constant_target"=> 1.0,
        "exploring_starts"=>true,
    )


function construct_agent(parsed)
    
end

function get_horde(parsed)

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


