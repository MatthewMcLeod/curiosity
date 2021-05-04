module TwoDGridWorldEvalSet

using Curiosity
using Curiosity: OpenWorld
using MinimalRLCore
const TDGWU = Curiosity.TwoDGridWorldUtils
include("../experiments/2d-gridworld.jl")
using Random
using StatsBase
using GVFHordes
using Statistics
using JLD2

using ProgressMeter

StatsBase.sample(p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(Weights([p.func(;state_t = s, action_t = a) for a in actions]))

StatsBase.sample(rng, p::GVFHordes.GVFParamFuncs.AbstractPolicy, s, actions) =
    sample(Weights([get(p;state_t = s, action_t = a) for a in actions]))

function gen_dataset(num_start_states=500)
    parsed = TwoDGridWorldExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"

    horde = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(i+2),
             TDGWU.GoalTermination(0.95),
             TDGWU.GoalPolicy(i, true)) for i in 1:4])

    cumulant_schedule = TDGWU.get_cumulant_schedule(parsed)

    parsed["exploring_starts"] = "whole"
    parsed["env_step_penalty"] = -0.005
    env = OpenWorld(10, 10,
                    cumulant_schedule=cumulant_schedule,
                    start_type=:none, normalized=true)

    start_states = Vector{Float64}[]

    for i in 1:num_start_states
        MinimalRLCore.reset!(env)
        s = MinimalRLCore.get_state(env)
        push!(start_states, s[1:2])
    end
    # @show start_states
    num_returns = 1000
    γ_thresh=1e-6
    horde_rets = zeros(length(horde.gvfs), length(start_states))
    actions = rand(1:4, length(start_states))
    # actions = fill(1, length(start_states))
    for (gvf_i,gvf) in enumerate(horde.gvfs)
        rets = monte_carlo_returns_agg(env, gvf, start_states, actions, num_returns, γ_thresh; agg=mean)
        horde_rets[gvf_i, :] = rets
    end

    return horde_rets, start_states, actions
end

function save_data(num_start_states=500)
    rets = gen_dataset(num_start_states)
    eval_set = Dict(
        "ests" => rets[1],
        "actions" => rets[3],
        "states" => rets[2]
    )
    @save "./src/data/TwoDGridWorldSet.jld2" eval_set
end

end
