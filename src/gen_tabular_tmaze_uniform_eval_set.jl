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
    # ind_to_delete = [21, 17, 5, 1]
    # for i in ind_to_delete
    #     start_states = del!(start_states, i)
    # end
    observations = []
    actions = []
    total_start_states = []
    for s in start_states
        MinimalRLCore.reset!(env,s)
        for a in get_actions(env)
            push!(observations, MinimalRLCore.get_state(env))
            push!(actions, a)
            push!(total_start_states, s)
        end
    end
    num_returns = 2
    γ_thresh=1e-6
    horde_rets = zeros(length(horde.gvfs), length(observations))
    for (gvf_i,gvf) in enumerate(horde.gvfs)
        rets = monte_carlo_returns(env, gvf, total_start_states, actions, num_returns, γ_thresh)
        rets_avg = mean(hcat(rets...),dims=1) # stack horizontally and then average over runs
        horde_rets[gvf_i,:] = rets_avg
    end

    # For tabular tmaze, the goal indices are implemented in such a way
    #the MC rollout doesnot trigger end episode until 1 step later.
    ind_to_zero = []
    for ind in 1:length(observations)
        if sum(observations[ind][2:end]) != 0.0
            push!(ind_to_zero, ind)
        end
    end
    for ind in ind_to_zero
        horde_rets[:,ind] .= 0.0
    end
    return horde_rets, observations, actions
end

function save_data(rets,obs,actions)
    TTMazeUniformEvalSet = Dict()
    TTMazeUniformEvalSet["ests"] = rets
    TTMazeUniformEvalSet["actions"] = actions
    TTMazeUniformEvalSet["states"] = obs
    @save "./src/data/TTMazeUniformEvalSet.jld2" TTMazeUniformEvalSet
end

end
