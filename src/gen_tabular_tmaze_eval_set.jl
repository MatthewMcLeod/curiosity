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

    horde = TabularTMazeExperiment.get_horde(parsed)

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
    for s in start_states
        MinimalRLCore.reset!(env,s)
        push!(observations, MinimalRLCore.get_state(env))
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

end
