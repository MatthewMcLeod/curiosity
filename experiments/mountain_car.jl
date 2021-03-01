


module MountainCarExperiment

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays

const MCU = Curiosity.MountainCarUtils

default_args() =
    Dict(
        "steps" => 50000,
        "seed" => 1,

        #Tile coding params used by Rich textbook for mountain car
        "numtilings" => 8,
        "numtiles" => 8,
        "behaviour_alpha" => 0.5/8,

        "behaviour_learner" => "ESARSA",
        "behaviour_rew" => "env",
        "behaviour_gamma" => 0.99,
        "intrinsic_reward" =>"no_reward",
        "behaviour_trace" => "replacing",

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
        observation_size = 4
        action_space = 3
        lambda = parsed["lambda"]
        demon_alpha = parsed["demon_alpha"]
        demon_alpha_init = parsed["demon_alpha_init"]
        demon_learner = parsed["demon_learner"]
        behaviour_learner = parsed["behaviour_learner"]
        behaviour_alpha = parsed["behaviour_alpha"]
        behaviour_gamma = parsed["behaviour_gamma"]
        behaviour_trace = parsed["behaviour_trace"]
        intrinsic_reward_type = parsed["intrinsic_reward"]

        #Create state constructor
        state_constructor_tc = TileCoder(parsed["numtilings"],parsed["numtiles"],observation_size)
        feature_size = size(state_constructor_tc)

        demons = get_horde(parsed)

        if demon_learner == "TB"
            demon_learner = TB(lambda, feature_size, length(demons), action_space, demon_alpha)
        elseif demon_learner == "TBAuto"
            demon_learner = TBAuto(lambda, feature_size, length(demons), action_space, demon_alpha, demon_alpha_init)
        else
            throw(ArgumentError("Not a valid demon learner"))
        end

        if behaviour_learner == "ESARSA"
            behaviour_learner = ESARSA(lambda, feature_size, 1, action_space, behaviour_alpha, behaviour_trace)
        else
            throw(ArgumentError("Not a valid behaviour learner"))
        end

        function state_constructor(obs, feature_size, tc)
            s = spzeros(feature_size)
            s[tc(obs)] .= 1
            return s
        end

        agent = Agent(demons, feature_size, feature_size, observation_size, action_space, demon_learner, behaviour_learner, intrinsic_reward_type, (obs) -> state_constructor(obs, feature_size, state_constructor_tc), behaviour_gamma)
end

function get_horde(parsed)
    action_space = 3
    return Horde([MCU.steps_to_wall_gvf(), MCU.steps_to_goal_gvf()])
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    seed = parsed["seed"]

    normalized = true
    env = MountainCar(normalized)

    agent = construct_agent(parsed)

    eps = 1
    max_num_steps = num_steps
    steps = Int[]

    while sum(steps) < max_num_steps
        cur_step = 0
        tr, stp =
            run_episode!(env, agent) do (s, a, s_next, r)
                #This is a callback for every timestep where logger can go
                # agent is accesible in this scope
                cur_step+=1
                if cur_step % 500 == 0
                    println("At step: ", cur_step)
                end
            end
            println("Finished episode: ", cur_step)
        push!(steps, stp)
        eps += 1
    end
end

end
