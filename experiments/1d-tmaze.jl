module OneDTmazeExperiment


import Flux: Descent
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
        "epsilon" => 0.2,
        "exploration_strategy" => "epsilon_greedy",

        # Demon Attributes
        "demon_alpha_init" => 0.1,
        "demon_eta" => 0.1/8,
        "demon_discounts" => 0.9,
        "demon_learner" => "SR",
        "demon_update" => "TB",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Descent",
        "demon_lambda" => 0.9,
        "demon_trace"=> "AccumulatingTraces",

        #shared
        "num_tiles" => 4,
        "num_tilings" =>20,
        "demon_num_tiles" => 4,
        "demon_num_tilings" => 8,

        # Environment Config
        "constant_target"=> 1.0,
        "cumulant"=>1.0,
        "cumulant_schedule" => "Constant",
        "distractor" => (1.0, 1.0),
        "drifter" => (1.0, sqrt(0.01)),
        "exploring_starts"=>"beg",

        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        # "logger_keys" => [LoggerKey.TTMAZE_ERROR],
        "save_dir" => "OneDTMazeExperiment",
        "seed" => 1,
        "steps" => 4000,
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

    demon_projected_fc = Curiosity.FeatureSubset(
        Curiosity.SparseTileCoder(parsed["demon_num_tilings"], parsed["demon_num_tiles"], 2),
        1:2)

    # demons = get_horde(parsed, length(demon_feature_projector), action_space, demon_feature_projector)
    # Lets make a simple horde

    demons = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(i+2),
             ODTMU.GoalTermination(0.9),
             ODTMU.GoalPolicy(i)) for i in 1:4])

     SF_horde = SRCU.get_SF_horde_for_policy(ODTMU.GoalPolicy(1), ODTMU.GoalTermination(0.9),demon_projected_fc,1:action_space)
     for i in 2:4
         SF_horde = Curiosity.GVFSRHordes.merge(SF_horde,SRCU.get_SF_horde_for_policy(ODTMU.GoalPolicy(i), ODTMU.GoalTermination(0.9),demon_projected_fc,1:action_space))
     end
     num_SFs = 4
     demons = Curiosity.GVFSRHordes.SRHorde(demons, SF_horde, num_SFs, demon_projected_fc)

    exploration_strategy = if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["epsilon"])
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end

    # demon_learner = Curiosity.get_linear_learner(parsed,
    #                                              feat_size,
    #                                              action_space,
    #                                              demons,
    #                                              "demon",
    #                                              nothing)
     demon_learner = Curiosity.get_linear_learner(parsed,
                                                  feat_size,
                                                  action_space,
                                                  demons,
                                                  "demon",
                                                  demon_projected_fc)

    # behaviour_learner = LinearQLearning()
    behaviour_learner = ODTMU.RoundRobinPolicy()




    # TODO: Behaviour horde needs access to the behaviour learner to condition the behaviour policy
    # BUT behaviour learner needs access the horde to know things like how many demons there are.
    # behaviour_num_tasks = 1
    # num_SFs = 4
    # num_demons = if parsed["behaviour_learner"] ∈ ["GPI"]
    #     num_SFs * feat_size * action_space + 1
    # elseif parsed["behaviour_learner"] ∈ ["Q"]
    #     1
    # end

    # behaviour_learner = Curiosity.get_linear_learner(parsed,
    #                                                  feat_size,
    #                                                  action_space,
    #                                                  num_demons,
    #                                                  behaviour_num_tasks,
    #                                                  "behaviour",
    #                                                  nothing)

    # behaviour_gvf = TTMU.make_behaviour_gvf(behaviour_discount, state_constructor_func, behaviour_learner, exploration_strategy)
    # behaviour_demons = if behaviour_learner isa GPI
    #     SF_horde = TTMU.make_SF_horde(behaviour_discount, feature_size, action_space, behaviour_feature_projector)

    #     pred_horde = Horde([behaviour_gvf])

    #     Curiosity.GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, behaviour_feature_projector)
    # elseif behaviour_learner isa QLearner
    #     Horde([behaviour_gvf])
    # else
    #     throw(ArgumentError("goes with which horde? " ))
    # end


    # Agent(demons,
    #       feature_size,
    #       behaviour_learner,
    #       behaviour_demons,
    #       behaviour_discount,
    #       demon_learner,
    #       observation_size,
    #       action_space,
    #       intrinsic_reward_type,
    #       (obs) -> state_constructor(obs, feature_size),
    #       use_external_reward,
    #       exploration_strategy)

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

# function get_horde(parsed, feature_size, action_space, projected_feature_constructor)

#     discount = parsed["demon_discounts"]
#     pseudoterm = TTMU.pseudoterm
#     num_actions = TTMU.NUM_ACTIONS
#     num_demons = TTMU.NUM_DEMONS


#     #TODO: Sort out the if-else block so that demon_policy_type and horde_type is not blocking eachother.
#     horde = if parsed["demon_policy_type"] == "greedy_to_cumulant" && parsed["horde_type"] == "regular"
#         Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.FunctionalPolicy((;kwargs...) -> TTMU.demon_target_policy(i;kwargs...))) for i in 1:num_demons])
#     elseif parsed["demon_policy_type"] == "random" && parsed["horde_type"] == "regular"
#         Horde([GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions))) for i in 1:num_demons])
#     else
#         throw(ArgumentError("Not a valid policy type for demons"))
#     end

#     if parsed["demon_learner"] == "SR"
#         num_SFs = 4
#         SF_horde = TTMU.make_SF_horde(discount, length(projected_feature_constructor), action_space, projected_feature_constructor)

#         horde = Curiosity.GVFSRHordes.SRHorde(horde, SF_horde, num_SFs, projected_feature_constructor)
#     end

#     return horde
# end

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
                        goals = s_next[2:end]
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
        agent
    end

end


end
