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
        "numtilings" => 8,
        "numtiles" => 8,


        "behaviour_update" => "ESARSA",
        "behaviour_learner" => "Q",
        "behaviour_eta" => 0.5/8,
        "behaviour_opt" => "Descent",
        # "behaviour_rew" => "env",
        "behaviour_gamma" => 0.99,
        "behaviour_lambda" => 0.9,

        "intrinsic_reward" =>"no_reward",
        "behaviour_trace" => "replacing",
        "use_external_reward" => true,
        "exploration_strategy" => "epsilon_greedy",
        "exploration_param" => 0.2,

        "lambda" => 0.9,
        "demon_alpha" => 0.1/8,
        "demon_alpha_init" => 0.1/8,
        "demon_learner" => "Q",
        "demon_update" => "ESARSA",
        "demon_opt" => "Auto",
        "demon_lambda" => 0.9,
        "exploring_starts"=>true,
        "save_dir" => "MountainCarExperiment",
        "logger_keys" => [LoggerKey.EPISODE_LENGTH, LoggerKey.MC_ERROR],

    )


function construct_agent(parsed)
    observation_size = 2
    action_space = 3

    behaviour_learner = parsed["behaviour_learner"]
    behaviour_lu = parsed["behaviour_update"]

    behaviour_gamma = parsed["behaviour_gamma"]
    behaviour_trace = parsed["behaviour_trace"]
    intrinsic_reward_type = parsed["intrinsic_reward"]
    use_external_reward = parsed["use_external_reward"]

    #Create state constructor
    state_constructor_tc =
        TileCoder(parsed["numtilings"], parsed["numtiles"], observation_size)

    feature_size = size(state_constructor_tc)
    function state_constructor(obs, feature_size, tc)
        s = spzeros(Int, feature_size)
        s[tc(obs)] .= 1
        return s
    end

    demons = get_horde(parsed,
                       feature_size,
                       action_space,
                       (obs) -> state_constructor(obs,
                                                  feature_size,
                                                  state_constructor_tc))

    demon_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 demons,
                                                 "demon")

    behaviour_demons = if behaviour_learner ∈ ["GPI"]
        get_GPI_horde(parsed,
                           feature_size,
                           action_space, (obs) ->
                           state_constructor(obs, feature_size, state_constructor_tc))
    else
        nothing
    end

    behaviour_learner = Curiosity.get_linear_learner(parsed,
                                                     feature_size,
                                                     action_space,
                                                     behaviour_demons,
                                                     "behaviour")

    # behaviour_lu = if behaviour_lu == "ESARSA"
    #     ESARSA(lambda=parsed["lambda"], opt=Descent(behaviour_alpha))
    # elseif behaviour_lu == "SARSA"
    #     SARSA(lambda=parsed["lambda"], opt=Descent(behaviour_alpha))
    # elseif behaviour_lu == "TB"
    #     TB(lambda=parsed["lambda"], opt=Descent(behaviour_alpha))
    # else
    #     throw(ArgumentError("$(behaviour_lu) Not a valid behaviour learning update"))
    # end

    # behaviour_learner = if behaviour_learner ∈ ["Q", "QLearner", "q"]
    #     LinearQLearner(behaviour_lu, feature_size, action_space, 1)
    # elseif behaviour_learner ∈ ["GPI"]
    #     GPI(behaviour_lu,
    #               feature_size,
    #               length(behaviour_demons),
    #               action_space,
    #               behaviour_demons.num_tasks)
    # else
    #     throw(ArgumentError("$(behaviour_learner) Not a valid behaviour learner"))
    # end


    exploration_strategy = if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["exploration_param"])
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end

    behaviour_gvf = MCU.make_behaviour_gvf(behaviour_gamma, (obs) -> state_constructor(obs, feature_size, state_constructor_tc), behaviour_learner, exploration_strategy)
    behaviour_demons = if behaviour_learner isa GPI
        SF_horde = TTMU.make_SF_horde(behaviour_discount, feature_size, action_space)
        num_SFs = 4

        pred_horde = Horde([behaviour_gvf])

        Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, (obs) -> state_constructor(obs, feature_size, state_constructor_tc))
    elseif behaviour_learner isa QLearner
        Horde([behaviour_gvf])
    end

    Agent(demons,
          feature_size,
          behaviour_learner,
          behaviour_demons,
          behaviour_gamma,
          demon_learner,
          observation_size,
          action_space,
          intrinsic_reward_type,
          (obs) -> state_constructor(obs, feature_size, state_constructor_tc),
          use_external_reward,
          exploration_strategy)

end

function get_GPI_horde(parsed, feature_size, action_space, state_constructor)
    discount = parsed["behaviour_gamma"]
    SF_horde = MCU.make_SF_horde(feature_size, action_space, state_constructor)
    num_SFs = 2
    #NOTE: Tasks is learning the reward feature vector
    #Dummy prediction GVF
    DummyGVF = GVF(GVFParamFuncs.FeatureCumulant(1), GVFParamFuncs.ConstantDiscount(0.0), GVFParamFuncs.NullPolicy())
    pred_horde = Horde([DummyGVF])

    return Curiosity.GVFSRHordes.SRHorde(pred_horde,
        SF_horde,
        num_SFs,
        state_constructor)
end

function get_horde(parsed, feature_size, action_space, state_constructor)
    horde = Horde([MCU.steps_to_wall_gvf(), MCU.steps_to_goal_gvf()])
    if parsed["demon_learner"] == "SR"
         SF_horde = MCU.make_SF_horde(feature_size, action_space, state_constructor)
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

            max_episode_steps = min(max_num_steps - sum(steps), 1000)
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
