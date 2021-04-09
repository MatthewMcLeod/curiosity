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
        "steps" => 30000,
        "seed" => 1,

        #Tile coding params used by Rich textbook for mountain car
        "numtilings" => 4,
        "numtiles" => 20,
        "numtilings" => 3,
        "numtiles" => 8,
        "behaviourtilings" => 3,
        "behaviourtiles" => 7,
        "demontilings" => 8,
        "demontiles" => 4,


        "behaviour_update" => "ESARSA",
        "behaviour_learner" => "Q",
        "behaviour_eta" => 0.5/8,
        "behaviour_opt" => "Descent",
        # "behaviour_rew" => "env",
        "behaviour_gamma" => 0.99,
        "behaviour_lambda" => 0.9,

        "intrinsic_reward" =>"no_reward",
        "behaviour_trace" => "ReplacingTraces",
        "use_external_reward" => true,
        "exploration_strategy" => "epsilon_greedy",
        "exploration_param" => 0.2,

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

    behaviour_learner = parsed["behaviour_learner"]
    behaviour_lu = parsed["behaviour_update"]

    behaviour_gamma = parsed["behaviour_gamma"]
    behaviour_trace = parsed["behaviour_trace"]
    intrinsic_reward_type = parsed["intrinsic_reward"]
    use_external_reward = parsed["use_external_reward"]

    #Create state constructor
    state_constructor_tc =
        TileCoder(parsed["numtilings"], parsed["numtiles"], observation_size)

    demon_tc =
        TileCoder(parsed["demontilings"], parsed["demontiles"], observation_size)
    behaviour_tc =
        TileCoder(parsed["behaviourtilings"], parsed["behaviourtiles"], observation_size)

    feature_size = size(state_constructor_tc)
    demon_feature_size = size(demon_tc)
    behaviour_feature_size = size(behaviour_tc)

    function state_constructor(obs, tc)
        s = spzeros(Int, size(tc))
        s[tc(obs)] .= 1
        return s
    end

    base_state_constructor_func = (obs) -> state_constructor(obs,
                               state_constructor_tc)
    demon_state_constructor_func = (obs) -> state_constructor(obs, demon_tc)
    behaviour_state_constructor_func = (obs) -> state_constructor(obs, behaviour_tc)

    demon_feature_projector = ActionValueFeatureProjector(demon_state_constructor_func, size(demon_tc))
    behaviour_feature_projector = ActionValueFeatureProjector(behaviour_state_constructor_func, size(behaviour_tc))


    # demons = get_horde(parsed,
    #                    feature_size,
    #                    action_space,
    #                    (obs) -> state_constructor(obs,
    #                                               feature_size,
    #                                               state_constructor_tc))
      demons = get_horde(parsed,
                         feature_size,
                         action_space,
                         demon_feature_projector)

    demon_learner = Curiosity.get_linear_learner(parsed,
                                                 feature_size,
                                                 action_space,
                                                 demons,
                                                 "demon",
                                                 demon_feature_projector)

     behaviour_num_tasks = 1
     num_SFs = 2
     num_demons = if parsed["behaviour_learner"] ∈ ["GPI"]
         num_SFs * length(behaviour_feature_projector) * action_space + 1
     elseif parsed["behaviour_learner"] ∈ ["Q"]
         1
    else
        throw(ArgumentError("Hacky thing not working"))
     end

     behaviour_learner = Curiosity.get_linear_learner(parsed,
                                                      feature_size,
                                                      action_space,
                                                      num_demons,
                                                      behaviour_num_tasks,
                                                      "behaviour",
                                                      behaviour_feature_projector)


    exploration_strategy = if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["exploration_param"])
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end

    behaviour_gvf = MCU.make_behaviour_gvf(behaviour_gamma, base_state_constructor_func, behaviour_learner, exploration_strategy)
    behaviour_demons = if behaviour_learner isa GPI
        SF_horde = MCU.make_SF_horde(behaviour_gamma, length(behaviour_feature_projector), action_space, behaviour_feature_projector)

        pred_horde = Horde([behaviour_gvf])

        Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, behaviour_feature_projector)
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
          base_state_constructor_func,
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
