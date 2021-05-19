module MountainCarExperiment

import Random

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays
using ProgressMeter

import Flux: Descent

const MCU = Curiosity.MountainCarUtils

default_args() =
    Dict(
        "steps" => 50000,
        "seed" => 1,

        #Tile coding params used by Rich textbook for mountain car
        "num_tilings" => 3,
        "num_tiles" => 8,
        "behaviour_num_tilings" => 8,
        "behaviour_num_tiles" => 2,
        "demon_num_tilings" => 8,
        "demon_num_tiles" => 4,


        "behaviour_update" => "ESARSA",
        "behaviour_learner" => "Q",
        "behaviour_eta" => 0.5/8,
        "behaviour_opt" => "Descent",
        # "behaviour_rew" => "env",
        "behaviour_gamma" => 0.99,
        "behaviour_lambda" => 0.95,
        "behaviour_w_init" => 0.0,

        "intrinsic_reward" =>"no_reward",
        "behaviour_trace" => "ReplacingTraces",
        "use_external_reward" => true,
        "exploration_strategy" => "epsilon_greedy",
        "exploration_param" => 0.2,
        "random_first_action" => false,

        "lambda" => 0.0,
        "demon_eta" => 0.1/8,
        "demon_alpha_init" => 0.1/8,
        "demon_learner" => "Q",
        "demon_update" => "TB",
        "demon_opt" => "Descent",
        "demon_lambda" => 0.9,
        "exploring_starts"=>true,
        "save_dir" => "MountainCarExperiment",
        "logger_keys" => [LoggerKey.EPISODE_LENGTH, LoggerKey.MC_ERROR],

    )


function construct_agent(parsed)
    observation_size = 2
    action_space = 3

    if "eta" in keys(parsed)
        prefixes = ["behaviour","demon"]
        for prefix in prefixes
            parsed[join([prefix, "eta"], "_")] = parsed["eta"]
        end
    end
    if parsed["demon_opt"] == "Descent"
        if "demon_eta" in keys(parsed)
            parsed["demon_eta"] = parsed["demon_eta"] / parsed["num_tilings"]
        end
    end
    if parsed["behaviour_opt"] == "Descent"
        if "behaviour_eta" in keys(parsed)
            parsed["behaviour_eta"] = parsed["behaviour_eta"] / parsed["num_tilings"]
        end
    end

    behaviour_learner = parsed["behaviour_learner"]
    behaviour_lu = parsed["behaviour_update"]

    behaviour_gamma = parsed["behaviour_gamma"]
    behaviour_trace = parsed["behaviour_trace"]
    intrinsic_reward_type = parsed["intrinsic_reward"]
    use_external_reward = parsed["use_external_reward"]


    fc = Curiosity.FeatureSubset(
        Curiosity.SparseTileCoder(parsed["num_tilings"], parsed["num_tiles"], 2),
        1:2)

    demon_reward_features = Curiosity.FeatureProjector(Curiosity.FeatureSubset(
                    Curiosity.SparseTileCoder(parsed["demon_num_tilings"], parsed["demon_num_tiles"], 2),
                1:2), false)


    behaviour_reward_features =  Curiosity.FeatureProjector(Curiosity.FeatureSubset(
                    Curiosity.SparseTileCoder(parsed["behaviour_num_tilings"], parsed["behaviour_num_tiles"], 2),
                1:2), false)

    feature_size = size(fc)
    demon_feature_size = size(demon_reward_features)
    behaviour_feature_size = size(behaviour_reward_features)

    demons = MCU.create_demons(parsed,
                         demon_reward_features)

    demon_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 demons,
                                                 "demon",
                                                 demon_reward_features)

    exploration_strategy = Curiosity.get_exploration_strategy(parsed, 1:action_space)

    behaviour_num_tasks = 1
    num_SFs = 4
    num_demons = if parsed["behaviour_learner"] ∈ ["GPI"]
        # num_SFs * size(behaviour_reward_projector) * action_space + behaviour_num_tasks
        num_SFs * size(behaviour_reward_projector) + behaviour_num_tasks

    elseif parsed["behaviour_learner"] ∈ ["Q"]
        behaviour_num_tasks
    elseif parsed["behaviour_learner"] == "RoundRobin"
        0
    else
        throw(ArgumentError("Hacky thing not working, yay!"))
    end
    behaviour_learner = Curiosity.get_linear_learner(parsed,
                                                     size(fc),
                                                     action_space,
                                                     num_demons,
                                                     behaviour_num_tasks,
                                                     "behaviour",
                                                     behaviour_reward_features)



    behaviour_demons = if behaviour_learner isa GPI
        @assert !(behaviour_reward_projector isa Nothing)
        bh_gvf = MCU.make_behaviour_gvf(behaviour_learner,
                                          0.0,
                                          fc,
                                          exploration_strategy)
        pred_horde = GVFHordes.Horde([bh_gvf])
        SF_policies = [MCU.steps_to_wall_gvf().policy, MCU.steps_to_goal_gvf().policy]
        SF_discounts = [MCU.steps_to_wall_gvf().discount, MCU.steps_to_goal_gvf().discount]
        num_SFs = length(SF_policies)
        SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, behaviour_reward_features, 1:action_space)
        Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, behaviour_reward_features)
    elseif behaviour_learner isa QLearner
        bh_gvf = MCU.make_behaviour_gvf(behaviour_learner,
                                          parsed["behaviour_gamma"],
                                          fc,
                                          exploration_strategy)
        GVFHordes.Horde([bh_gvf])
    else
        throw(ArgumentError("goes with which horde? " ))
    end

    random_first_action = parsed["random_first_action"]
    Agent(demons,
          feature_size,
          behaviour_learner,
          behaviour_demons,
          behaviour_gamma,
          demon_learner,
          observation_size,
          action_space,
          intrinsic_reward_type,
          fc,
          use_external_reward,
          exploration_strategy,
          random_first_action)
end

function get_horde(parsed, feature_size, action_space, state_constructor)
    horde = Horde([MCU.steps_to_wall_gvf(), MCU.steps_to_goal_gvf()])
    if parsed["demon_learner"] == "SR"
         SF_horde = MCU.make_SF_horde(parsed["behaviour_gamma"], length(state_constructor), action_space, state_constructor)
         num_SFs = 2
         horde = Curiosity.GVFSRHordes.SRHorde(horde, SF_horde, num_SFs, state_constructor)
    end
    return horde
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]

    Random.seed!(parsed["seed"])

    normalized = true
    env = MountainCar(0.0,0.0,normalized)

    agent = construct_agent(parsed)

    logger_init_dict = Dict(
        LoggerInitKey.TOTAL_STEPS => num_steps,
        LoggerInitKey.INTERVAL => 1000,
    )

    Curiosity.experiment_wrapper(parsed, logger_init_dict, working) do parsed, logger
        eps = 1
        max_num_steps = num_steps
        steps = Int[]

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        while sum(steps) < max_num_steps
            is_terminal = false

            max_episode_steps = min(max_num_steps - sum(steps), 2000)
            tr, stp =
                run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
                    logger_step!(logger, env, agent, s, a, s_next, r, t)
                    if progress
                        next!(prg_bar)
                    end
                end
            logger_episode_end!(logger)

            push!(steps, stp)
            eps += 1
        end
        logger
    end
end

end
