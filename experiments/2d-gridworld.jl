module TwoDGridWorldExperiment

import Flux: Descent, ADAM
import Random

using GVFHordes
using Curiosity
using Curiosity: OpenWorld
using MinimalRLCore
using SparseArrays
using ProgressMeter



# const TTMU = Curiosity.TabularTMazeUtils
const TDGWU = Curiosity.TwoDGridWorldUtils
const SRCU = Curiosity.SRCreationUtils

default_args() =
    Dict(
        "logger_interval" => 100,
        "start_dist" => "center",

        # Behaviour Items
        "behaviour_eta" => 0.1/8,
        "behaviour_gamma" => 0.0,
        "behaviour_learner" => "GPI",
        "behaviour_update" => "TB",
        "behaviour_reward_projector" => "tilecoding",
        "behaviour_rp_tilings" => 2,
        "behaviour_rp_tiles" => 2,
        "behaviour_trace" => "AccumulatingTraces",
        "behaviour_opt" => "Descent",
        "behaviour_lambda" => 0.9,
        "behaviour_w_init" => 4.0,
        "exploration_param" => 0.2,
        "exploration_strategy" => "epsilon_greedy",
        "ϵ_range" => (0.4,0.1),
        "decay_period" => 5000,
        "warmup_steps" => 1000,


        # Demon Attributes
        "demon_alpha_init" => 0.1,
        "demon_eta" => 0.1/8,
        "demon_discounts" => 0.9,
        "demon_gamma" => 0.9,
        "demon_learner" => "SR",
        "demon_update" => "TB",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Auto",
        "demon_lambda" => 0.9,
        "demon_trace"=> "AccumulatingTraces",
        "demon_beta_m" => 0.99,
        "demon_beta_v" => 0.99,

        #shared
        "num_tiles" => 4,
        "num_tilings" => 8,
        "demon_rep" => "ideal",
        "demon_num_tiles" => 6,
        "demon_num_tilings" => 1,

        # Environment Config
        "constant_target"=> 1.0,
        "cumulant"=>1.0,
        "cumulant_schedule" => "Constant",
        "distractor" => (1.0, 1.0),
        "drifter" => (1.0, sqrt(0.01)),
        "exploring_starts"=>"whole",

        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        # "logger_keys" => [LoggerKey.TTMAZE_ERROR],
        "save_dir" => "TwoDGridWorldExperiment",
        "seed" => 1,
        "steps" => 10000,
        "use_external_reward" => true,

        "logger_keys" => ["TWODGRIDWORLDERROR", "TWODGRIDWORLDERRORDPI", "ONED_GOAL_VISITATION", "EPISODE_LENGTH", "INTRINSIC_REWARD", "BEHAVIOUR_ACTION_VALUES"]
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

    if parsed["behaviour_w_init"] != 0.0
        parsed["behaviour_w_init"] = parsed["behaviour_w_init"] / parsed["num_tilings"]
    end


    if "eta" in keys(parsed)
        prefixes = ["behaviour","demon"]
        for prefix in prefixes
            parsed[join([prefix, "eta"], "_")] = parsed["eta"]
        end
    end

    if "alpha_init" in keys(parsed)
        prefixes = ["behaviour", "demon"]
        for prefix in prefixes
            parsed[join([prefix, "alpha_init"], "_")] = parsed["alpha_init"]
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


    fc = Curiosity.FeatureSubset(
        Curiosity.SparseTileCoder(parsed["num_tilings"], parsed["num_tiles"], 2),
        1:2)
    feat_size = size(fc)

    demon_projected_fc = if "demon_rep" ∉ keys(parsed)
        nothing
    elseif parsed["demon_rep"] == "tilecoding"
        Curiosity.ActionValueFeatureProjector(
        Curiosity.FeatureProjector(
            Curiosity.FeatureSubset(
                Curiosity.SparseTileCoder(
                parsed["demon_num_tilings"],
                parsed["demon_num_tiles"], 2),
            1:2),
        false),
        action_space)

    elseif parsed["demon_rep"] == "ideal_state_action"
        Curiosity.FeatureSubset(Curiosity.ActionValueFeatureProjector(TDGWU.IdealDemonFeatures(), action_space), 1:2)
    elseif parsed["demon_rep"] == "ideal"
        Curiosity.FeatureSubset(TDGWU.IdealDemonFeatures(), 1:2)
    elseif parsed["demon_rep"] == "ideal_martha"
        # ODTMU.IdealDemonFeatures()
        Curiosity.FeatureProjector(Curiosity.FeatureSubset(
            Curiosity.ActionValueFeatureProjector(TDGWU.MarthaIdealDemonFeatures(), action_space),
        1:2), false)
    else
        throw(ArgumentError("Not a valid demon projection rep for SR"))
    end

    demons = TDGWU.create_demons(parsed, demon_projected_fc)

    demon_learner = Curiosity.get_linear_learner(parsed,
                                                 feat_size,
                                                 action_space,
                                                 demons,
                                                 "demon",
                                                 demon_projected_fc)

    exploration_strategy = Curiosity.get_exploration_strategy(parsed, 1:action_space)

# <<<<<<< HEAD
# =======
#     behaviour_reward_projector = if "behaviour_reward_projector" ∉ keys(parsed)
#         nothing
#     elseif parsed["behaviour_reward_projector"] == "tilecoding"
#         Curiosity.ActionValueFeatureProjector(Curiosity.FeatureProjector(Curiosity.FeatureSubset(
#             Curiosity.SparseTileCoder(parsed["behaviour_rp_tilings"], parsed["behaviour_rp_tiles"], 2),
#             1:2), false),action_space)
#     elseif parsed["behaviour_reward_projector"] == "base"
#         Curiosity.FeatureProjector(fc, false)
#     elseif parsed["behaviour_reward_projector"] == "identity"
#         Curiosity.FeatureProjector(Curiosity.FeatureSubset(identity, 1:2), false)
#     elseif parsed["behaviour_reward_projector"] == "ideal_state_action"
#         Curiosity.FeatureSubset(TDGWU.IdealStateActionDemonFeatures(action_space), 1:2)
#     elseif parsed["behaviour_reward_projector"] == "ideal"
#         Curiosity.FeatureProjector(Curiosity.FeatureSubset(
#             TDGWU.IdealDemonFeatures(), 1:2), true)
#     elseif parsed["behaviour_reward_projector"] == "ideal_martha"
#         Curiosity.FeatureProjector(Curiosity.FeatureSubset(
#             TDGWU.MarthaIdealDemonFeatures(), 1:2), true)
#     else
#         throw(ArgumentError("Not a valid demon projection rep for SR"))
#     end

# >>>>>>> 3cf6c1fc34933118c5d04c1ca5106d9c05e6591e
    behaviour_learner, behaviour_demons, behaviour_discount = if parsed["behaviour_learner"] == "RoundRobin"
        # ODTMU.RoundRobinPolicy(), nothing, 0.0
        TDGWU.RoundRobinPolicy(), nothing, 0.0
        # throw("Round Robin not available")
    else

        brp_str = "behaviour_reward_projector" ∈ keys(parsed) ? parsed["behaviour_reward_projector"] : "nothing"
<<<<<<< HEAD
        
=======

>>>>>>> 42b74ac9bc0349d4b05a80a6e16897ac7283e63f
        behaviour_reward_projector = if brp_str == "nothing"
            nothing
        elseif brp_str == "tilecoding"
            Curiosity.ActionValueFeatureProjector(
                Curiosity.FeatureProjector(
                    Curiosity.FeatureSubset(
                        Curiosity.SparseTileCoder(
                            parsed["behaviour_rp_tilings"],
                            parsed["behaviour_rp_tiles"],
                            2),
                        1:2), false), action_space)
        elseif brp_str == "base"
            Curiosity.ActionValueFeatureProjector(Curiosity.FeatureProjector(fc, false), action_space)
        elseif brp_str == "state_agg"
            Curiosity.ActionValueFeatureProjector(
                Curiosity.FeatureProjector(
                    TDGWU.StateAggregation(), false),
                action_space)
        else
            throw(ArgumentError("Not a valid demon projection rep for GPI"))
        end


        behaviour_num_tasks = 1
        num_SFs = 4
        num_demons = if parsed["behaviour_learner"] ∈ ["GPI"]
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
                                                         behaviour_reward_projector)



        behaviour_demons = if behaviour_learner isa GPI
            @assert !(behaviour_reward_projector isa Nothing)
            bh_gvf = TDGWU.make_behaviour_gvf(behaviour_learner,
                                              0.0,
                                              fc,
                                              exploration_strategy)
            pred_horde = GVFHordes.Horde([bh_gvf])

            SF_policies = [TDGWU.NaiveGoalPolicy(i) for i in 1:4]
            SF_discounts = [TDGWU.GoalTermination(parsed["behaviour_gamma"]) for i in 1:4]
            num_SFs = length(SF_policies)
            SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, behaviour_reward_projector, 1:action_space)
            Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, behaviour_reward_projector)
        elseif behaviour_learner isa QLearner
            bh_gvf = TDGWU.make_behaviour_gvf(behaviour_learner,
                                              parsed["behaviour_gamma"],
                                              fc,
                                              exploration_strategy)
            GVFHordes.Horde([bh_gvf])
        # elseif behaviour_learner isa ODTMU.RoundRobinPolicy
        #     nothing
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
          exploration_strategy,
          false)
end

function main_experiment(parsed=default_args(); progress=false, working=false)

    num_steps = parsed["steps"]
    logger_init_dict = Dict(
        LoggerInitKey.TOTAL_STEPS => num_steps,
        LoggerInitKey.INTERVAL => parsed["logger_interval"],
        # LoggerInitKey.ENV => "tabular_tmaze"
    )

    Curiosity.experiment_wrapper(parsed, logger_init_dict, working) do parsed, logger


        Random.seed!(parsed["seed"])

        cumulant_schedule = TDGWU.get_cumulant_schedule(parsed)

        start_dist = Symbol(parsed["start_dist"])
        per_step_rew = get(parsed, "env_step_penalty", 0.0)
        env = OpenWorld(10, 10,
                        per_step_rew=per_step_rew,
                        cumulant_schedule=cumulant_schedule,
                        start_type=start_dist)

        agent = construct_agent(parsed)

        goal_visitations = zeros(4)
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
            println(goal_visitations)
        end
        env, agent
    end

end

end
