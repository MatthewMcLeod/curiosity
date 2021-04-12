module OneDTmazeExperiment


import Flux: Descent, ADAM
import Random

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays
using ProgressMeter



# const TTMU = Curiosity.TabularTMazeUtils
const ODTMU = Curiosity.OneDTMazeUtils
const SRCU = Curiosity.SRCreationUtils

default_args() =
    Dict(
        # Behaviour Items
        "behaviour_eta" => 0.2,
        "behaviour_gamma" => 0.9,
        "behaviour_learner" => "RoundRobin",
        "behaviour_update" => "none",
        "behaviour_trace" => "AccumulatingTraces",
        "behaviour_opt" => "Descent",
        "behaviour_lambda" => 0.9,
        "exploration_param" => 0.2,
        "exploration_strategy" => "epsilon_greedy",

        # Demon Attributes
        "demon_alpha_init" => 0.1,
        "demon_eta" => 0.1/8,
        "demon_discounts" => 0.9,
        "demon_learner" => "SR",
        "demon_update" => "TB",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Descent",
        "demon_lambda" => 0.0,
        "demon_trace"=> "AccumulatingTraces",

        #shared
        "num_tiles" => 4,
        "num_tilings" =>16,
        "demon_rep" => "tilecoding",
        "demon_num_tiles" => 4,
        "demon_num_tilings" => 4,

        # Environment Config
        "constant_target"=> 1.0,
        "cumulant"=>1.0,
        "cumulant_schedule" => "Constant",
        "distractor" => (1.0, 1.0),
        "drifter" => (sqrt(0.01), 1.0),
        "exploring_starts"=>"whole",

        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        # "logger_keys" => [LoggerKey.TTMAZE_ERROR],
        "save_dir" => "OneDTMazeExperiment",
        "seed" => 1,
        "steps" => 10000,
        "use_external_reward" => true,

        "logger_keys"=>[LoggerKey.ONEDTMAZEERROR]
    )


function construct_agent(parsed)


    action_space = 4
    obs_size = 6
    demon_alpha_init = parsed["demon_alpha_init"]
    demon_learner = parsed["demon_learner"]
    demon_lu = parsed["demon_update"]

    behaviour_learner = parsed["behaviour_learner"]
    behaviour_lu = parsed["behaviour_update"]
    behaviour_discount = parsed["behaviour_gamma"]

    intrinsic_reward_type = parsed["intrinsic_reward"]
    behaviour_trace = parsed["behaviour_trace"]
    use_external_reward = parsed["use_external_reward"]


    fc = Curiosity.FeatureSubset(
        Curiosity.SparseTileCoder(parsed["num_tilings"], parsed["num_tiles"], 2),
        1:2)
    feat_size = size(fc)

    demon_projected_fc = if parsed["demon_rep"] == "tilecoding"
        Curiosity.FeatureSubset(
        Curiosity.SparseTileCoder(parsed["demon_num_tilings"], parsed["demon_num_tiles"], 2),
        1:2)
    elseif parsed["demon_rep"] == "ideal"
        ODTMU.IdealDemonFeatures()
    else
        throw(ArgumentError("Not a valid demon projection rep for SR"))
    end


    demons = ODTMU.create_demons(parsed, demon_projected_fc)

    exploration_strategy = if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["exploration_param"])
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end

     demon_learner = Curiosity.get_linear_learner(parsed,
                                                  feat_size,
                                                  action_space,
                                                  demons,
                                                  "demon",
                                                  demon_projected_fc)

    behaviour_learner = ODTMU.RoundRobinPolicy()

    Agent(demons,
          feat_size,
          behaviour_learner,
          # behaviour_demons,
          nothing,
          behaviour_discount,
          demon_learner,
          obs_size,
          action_space,
          intrinsic_reward_type,
          fc,
          use_external_reward,
          exploration_strategy)
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    Random.seed!(parsed["seed"])

    cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)

    # exploring_starts = parsed["exploring_starts"]
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"])

    agent = construct_agent(parsed)

    goal_visitations = zeros(4)

    logger_init_dict = Dict(
        LoggerInitKey.TOTAL_STEPS => num_steps,
        LoggerInitKey.INTERVAL => 50,
        # LoggerInitKey.ENV => "tabular_tmaze"
    )

    Curiosity.experiment_wrapper(parsed, logger_init_dict, working) do parsed, logger
        eps = 1
        max_num_steps = num_steps
        steps = Int[]

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        while sum(steps) < max_num_steps
            cur_step = 0
            max_episode_steps = min(max_num_steps - sum(steps), 1000)
            tr, stp =
                run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
                    #This is a callback for every timestep where logger can go
                    # agent is accesible in this scope

                    if t == true && working==true
                        goals = s_next[3:end]
                        f = findfirst(!iszero, goals)
                        goal_visitations[f] += 1
                    end

                    if progress
                        next!(prg_bar)
                    end

                    logger_step!(logger, env, agent, s, a, s_next, r, t)
                    cur_step+=1
                end
                logger_episode_end!(logger)
            push!(steps, stp)
            eps += 1
        end
        if working == true
            # @show sum(agent.demon_learner.ψ)
            # @show sum(agent.demon_learner.r_w)
            println(goal_visitations)
        end
        agent
    end

end


end
