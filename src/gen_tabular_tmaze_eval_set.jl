module TabularTMazeEvalSet

using Curiosity
using MinimalRLCore
const TTMU = Curiosity.TabularTMazeUtils
include("../experiments/tabular_tmaze.jl")
using Random
using StatsBase
using GVFHordes
using Statistics
using JLD2


StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(s, a) for a in actions]))


function gen_dataset()
    parsed = TabularTMazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0

    horde = TabularTMazeExperiment.get_horde(parsed, 21, 4, nothing)

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)
    start_states = Curiosity.valid_state_mask()

    function del!(arr, ind_to_del)
        arr = arr[1:end .!= ind_to_del]
        return arr
    end
    #NOTE: This lets the agent start in terminal states. This is not normally possible.
    # Work backwards with index so you dont have to reindex due to shifting
    ind_to_delete = [21, 17, 5, 1]
    for i in ind_to_delete
        start_states = del!(start_states, i)
    end
    observations = []
    start_states_all = []
    for s in start_states
        MinimalRLCore.reset!(env,s)
        push!(observations, MinimalRLCore.get_state(env))
        push!(start_states_all, s)
    end
    total_observations = []
    total_start_states = []
    total_actions = []
    for (obs_ind,obs) in enumerate(observations)
            unique_actions = unique([StatsBase.sample(GVFHordes.policy(gvf), obs, get_actions(env)) for gvf in horde.gvfs])
            for ua in unique_actions
                push!(total_actions, ua)
                push!(total_observations, obs)
                push!(total_start_states, start_states[obs_ind])
            end
    end


    num_returns = 2
    γ_thresh=1e-6
    actions = zeros(Int,length(horde.gvfs), length(total_observations))
    horde_rets = zeros(length(horde.gvfs), length(total_observations))
    for (gvf_i,gvf) in enumerate(horde.gvfs)
        rets = monte_carlo_returns(env, gvf, total_start_states, total_actions, num_returns, γ_thresh)
        rets_avg = mean(hcat(rets...),dims=1) # stack horizontally and then average over runs
        horde_rets[gvf_i,:] = rets_avg
    end


    return horde_rets, total_observations, total_actions
end

function save_data(rets,obs,actions)
    TTMazeEvalSet = Dict()
    TTMazeEvalSet["ests"] = rets
    TTMazeEvalSet["actions"] = actions
    TTMazeEvalSet["states"] = obs
    @save "./src/data/TTMazeEvalSet.jld2" TTMazeEvalSet
end

function gen_roundrobin_dataset(num_start_states = 100)
    parsed = TabularTMazeExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0

    horde = TabularTMazeExperiment.get_horde(parsed, 21, 4, nothing)

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)

    exploring_starts = true
    env = TabularTMaze(exploring_starts, cumulant_schedule)


    all_states = []
    all_actions = []
    for gvf_i in 1:4
        # policy = gvf.policy
        policy = TTMU.GoalPolicy(gvf_i)
        for i in 1:num_start_states
            s = MinimalRLCore.start!(env)
            a = policy(s)
            push!(all_states, s)
            push!(all_actions, a)
            term = false
            while term == false
                s, r, term = MinimalRLCore.step!(env, a)
                if term
                    break
                end
                a = policy(s)
                push!(all_states, s)
                push!(all_actions, a)
            end
        end
    end
    eval_set_size = 500
    indices = sample(1:size(all_states)[1], eval_set_size, replace=false)
    ss_eval = all_states[indices]
    as_eval = all_actions[indices]
    num_returns = 2
    γ_thresh=1e-6

    horde_rets = zeros(length(horde.gvfs), length(ss_eval))

    for (gvf_i,gvf) in enumerate(horde.gvfs)
        ss_modified = [Int(s[1]) for s in ss_eval]
        rets = monte_carlo_returns(env, gvf, ss_modified, as_eval, num_returns, γ_thresh)
        rets_avg = mean(hcat(rets...),dims=1) # stack horizontally and then average over runs
        horde_rets[gvf_i,:] = rets_avg
    end
    return horde_rets, ss_eval, as_eval

end

function save_data_roundrobin(rets,obs,actions)
    TTMazeRoundRobinEvalSet = Dict()
    TTMazeRoundRobinEvalSet["ests"] = rets
    TTMazeRoundRobinEvalSet["actions"] = actions
    TTMazeRoundRobinEvalSet["states"] = obs
    @save "./src/data/TTMazeRoundRobinEvalSet.jld2" TTMazeRoundRobinEvalSet
end
end
