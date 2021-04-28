module OneDTmazeExperiment


import Flux: Descent, ADAM
import Random

using GVFHordes
using Curiosity
using MinimalRLCore
using SparseArrays
using ProgressMeter



const ODTMU = Curiosity.OneDTMazeUtils
const SRCU = Curiosity.SRCreationUtils

default_args() =
    Dict(
        "logger_interval" => 100,

        # Behaviour Items
        # "behaviour_eta" => 0.1/8,
        "behaviour_gamma" => 0.9,
        "behaviour_learner" => "GPI",
        "behaviour_update" => "TB",
        "behaviour_reward_projector" => "base",
        "behaviour_rp_tilings" => 1,
        "behaviour_rp_tiles" => 16,
        "behaviour_trace" => "ReplacingTraces",
        "behaviour_opt" => "Auto",
        "behaviour_lambda" => 0.95,
        "behaviour_alpha_init" => 0.1,
        "exploration_param" => 0.1,
        "exploration_strategy" => "epsilon_greedy",
        "ϵ_range" => (0.4,0.1),
        "decay_period" => 5000,
        "warmup_steps" => 12000,
        "behaviour_w_init" => 1,

        # Demon Attributes
        "demon_alpha_init" => 0.1,
        # "demon_eta" => 0.1/8,
        "demon_discounts" => 0.9,
        "demon_learner" => "NoLearner",
        "demon_update" => "TB",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Auto",
        "demon_lambda" => 0.9,
        "demon_trace"=> "AccumulatingTraces",
        "demon_beta_m" => 0.99,
        "demon_beta_v" => 0.99,

        #shared
        # "num_tiles" => 2,
        # "num_tilings" =>8,
        "tiling_structure" => [1,16],
        "demon_rep" => "ideal_martha",
        # "demon_rep" => "tilecoding",
        "demon_num_tiles" => 8,
        "demon_num_tilings" => 1,
        "eta" => 0.01,

        # Environment Config
        "constant_target"=> [-10.0,10.0],
        "cumulant_schedule" => "DrifterDistractor",
        "distractor" => (1.0, 5.0),
        "drifter" => (1.0, sqrt(0.01)),
        "exploring_starts"=>"whole",
        "env_step_penalty" => -0.1,


        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "no_reward",
        # "logger_keys" => [LoggerKey.TTMAZE_ERROR],
        "save_dir" => "OneDTMazeExperiment",
        "seed" => 1,
        "steps" => 100,
        "use_external_reward" => true,

        "logger_keys" => [LoggerKey.ONEDTMAZEERROR, LoggerKey.ONED_GOAL_VISITATION, LoggerKey.EPISODE_LENGTH, LoggerKey.INTRINSIC_REWARD, LoggerKey.BEHAVIOUR_ACTION_VALUES]
    )


function construct_agent(parsed)
    action_space = 4
    obs_size = 6

    intrinsic_reward_type = parsed["intrinsic_reward"]
    use_external_reward = parsed["use_external_reward"]

    if "tiling_structure" ∈ keys(parsed)
        parsed["num_tilings"] = parsed["tiling_structure"][1]
        parsed["num_tiles"] = parsed["tiling_structure"][2]
    end

    if "eta" in keys(parsed)
        prefixes = ["behaviour","demon"]
        for prefix in prefixes
            parsed[join([prefix, "eta"], "_")] = parsed["eta"]
        end
    end

    if parsed["demon_opt"] == "Auto"
        parsed["demon_alpha_init"] =
            parsed["demon_alpha_init"] / parsed["num_tilings"]
    end

    if "behaviour_opt" ∈ keys(parsed) && parsed["behaviour_opt"] == "Auto"
        parsed["behaviour_alpha_init"] =
            parsed["behaviour_alpha_init"] / parsed["num_tilings"]
    end

    if parsed["demon_opt"] == "Descent"
        if "eta" in keys(parsed)
            parsed["eta"] = parsed["eta"]/ parsed["num_tilings"]
        end
        if "demon_eta" in keys(parsed)
            parsed["demon_eta"] = parsed["demon_eta"] / parsed["num_tilings"]
        end
    end

    if parsed["behaviour_w_init"] != 0.0
        parsed["behaviour_w_init"] = parsed["behaviour_w_init"] / parsed["num_tilings"]
    end

    fc = Curiosity.FeatureSubset(
        Curiosity.SparseTileCoder(parsed["num_tilings"], parsed["num_tiles"], 2),
        1:2)
    feat_size = size(fc)

    demon_projected_fc = if "demon_rep" ∉ keys(parsed)
        nothing
    elseif parsed["demon_rep"] == "tilecoding"
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(
                Curiosity.SparseTileCoder(parsed["demon_num_tilings"], parsed["demon_num_tiles"], 2),
            1:2), false)
    elseif parsed["demon_rep"] == "ideal"
        # ODTMU.IdealDemonFeatures()
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(ODTMU.IdealDemonFeatures(), 1:2), true)
    elseif parsed["demon_rep"] == "ideal_martha"
        # ODTMU.IdealDemonFeatures()
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(ODTMU.MarthaIdealDemonFeatures(), 1:2), false)

    else
        throw(ArgumentError("Not a valid demon projection rep for SR"))
    end

    demons = ODTMU.create_demons(parsed, demon_projected_fc)


    demon_learner = Curiosity.get_linear_learner(parsed,
                                feat_size,
                                action_space,
                                demons,
                                "demon",
                                demon_projected_fc)

    exploration_strategy = Curiosity.get_exploration_strategy(parsed, 1:action_space)

    behaviour_reward_projector = if "behaviour_reward_projector" ∉ keys(parsed)
        nothing
    elseif parsed["behaviour_reward_projector"] == "tilecoding"
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(
            Curiosity.SparseTileCoder(parsed["behaviour_rp_tilings"], parsed["behaviour_rp_tiles"], 2),
            1:2), false)
    elseif parsed["behaviour_reward_projector"] == "base"
        Curiosity.FeatureProjector(fc, false)
    elseif parsed["behaviour_reward_projector"] == "identity"
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(identity, 1:2), false)
    elseif parsed["behaviour_reward_projector"] == "ideal"
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(
            ODTMU.IdealDemonFeatures(), 1:2), true)
    elseif parsed["behaviour_reward_projector"] == "ideal_martha"
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(
            ODTMU.MarthaIdealDemonFeatures(), 1:2), true)
    else
        throw(ArgumentError("Not a valid demon projection rep for SR"))
    end

    behaviour_learner, behaviour_demons, behaviour_discount = if parsed["behaviour_learner"] == "RoundRobin"
        ODTMU.RoundRobinPolicy(), nothing, 0.0
    else
        behaviour_num_tasks = 1
        num_SFs = 4
        num_demons = if parsed["behaviour_learner"] ∈ ["GPI"]
            num_SFs * size(behaviour_reward_projector) * action_space + behaviour_num_tasks
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
                                                         behaviour_reward_projector)



        behaviour_demons = if behaviour_learner isa GPI
            @assert !(behaviour_reward_projector isa Nothing)
            bh_gvf = ODTMU.make_behaviour_gvf(behaviour_learner,
                                              0.0,
                                              fc,
                                              exploration_strategy)
            pred_horde = GVFHordes.Horde([bh_gvf])
            SF_policies = [ODTMU.GoalPolicy(i) for i in 1:4]
            SF_discounts = [ODTMU.GoalTermination(parsed["behaviour_gamma"]) for i in 1:4]
            num_SFs = length(SF_policies)
            SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, behaviour_reward_projector, 1:action_space)
            Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, behaviour_reward_projector)
        elseif behaviour_learner isa QLearner
            bh_gvf = ODTMU.make_behaviour_gvf(behaviour_learner,
                                              parsed["behaviour_gamma"],
                                              fc,
                                              exploration_strategy)
            GVFHordes.Horde([bh_gvf])
        elseif behaviour_learner isa ODTMU.RoundRobinPolicy
            nothing
        else
            throw(ArgumentError("goes with which horde? " ))
        end

        behaviour_learner, behaviour_demons, parsed["behaviour_gamma"]
    end #end behaviour_learner, behaviour_demons, behaviour_discount = if parsed["behaviour_learner"] == "RoundRobin"

    Agent(demons,
          feat_size,
          behaviour_learner,
          behaviour_demons,
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
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"])

    agent = construct_agent(parsed)

    goal_visitations = zeros(4)

    logger_init_dict = Dict(
        LoggerInitKey.TOTAL_STEPS => num_steps,
        LoggerInitKey.INTERVAL => parsed["logger_interval"],
        # LoggerInitKey.ENV => "tabular_tmaze"
    )

    Curiosity.experiment_wrapper(parsed, logger_init_dict, working) do parsed, logger
        eps = 1
        max_num_steps = num_steps
        steps = Int[]

        logger_start!(logger, env, agent)

        prg_bar = ProgressMeter.Progress(num_steps, "Step: ")
        while sum(steps) < max_num_steps
            cur_step = 0
            max_episode_steps = max_num_steps - sum(steps)
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
            println("Goal Visitation: ", goal_visitations)
        end
        # agent
    end

end


end
