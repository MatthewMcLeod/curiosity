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

function gen_dataset(num_start_states=500, seed=1)
    Random.seed!(seed)
    parsed = TwoDGridWorldExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"

    horde = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(i+2),
             TDGWU.GoalTermination(0.95),
             TDGWU.GoalPolicy(i, true)) for i in 1:4])

    cumulant_schedule = TDGWU.get_cumulant_schedule(parsed)

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
        rets = Curiosity.monte_carlo_returns_agg(env, gvf, start_states, actions, num_returns, γ_thresh; agg=mean)
        horde_rets[gvf_i, :] = rets
    end

    return horde_rets, start_states, actions
end


function gen_dataset_special(num_start_states=500, seed=1)
    Random.seed!(seed)
    parsed = TwoDGridWorldExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"

    rets = Vector{Float64}[]
    start_states = Vector{Vector{Float64}}[]
    actions = Vector{Int}[]

    for gvf_i ∈ 1:4
        horde = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(gvf_i+2),
             TDGWU.GoalTermination(0.95),
             TDGWU.GoalPolicy(gvf_i, true))])
        
        cumulant_schedule = TDGWU.get_cumulant_schedule(parsed)
        
        env = OpenWorld(10, 10,
                        cumulant_schedule=cumulant_schedule,
                        start_type=:center, normalized=true)
        
        ss = Vector{Float64}[]
        as = Int[]
        policy = TDGWU.GoalPolicy(gvf_i, true)
        
        for i in 1:num_start_states
            # MinimalRLCore.reset!(env)
            # s = MinimalRLCore.get_state(env)
            s = MinimalRLCore.start!(env)
            a = policy(s)
            for t ∈ 1:rand(1:15)
                s, r, term = MinimalRLCore.step!(env, a)
                if term
                    break
                end
                a = policy(s)
            end
            push!(ss, s[1:2])
            push!(as, a)
        end
        # @show start_states
        num_returns = 1000
        γ_thresh=1e-6

        horde_rets = Curiosity.monte_carlo_returns_agg(env, horde.gvfs[1], ss, as, num_returns, γ_thresh; agg=mean)

        push!(rets, horde_rets)
        push!(start_states, ss)
        push!(actions, as)
    end
        
        return rets, start_states, actions
end


function save_data(num_start_states=500, seed=1)
    rets = gen_dataset(num_start_states)
    eval_set = Dict(
        "ests" => rets[1],
        "actions" => rets[3],
        "states" => rets[2]
    )
    @save "./src/data/TwoDGridWorldSet.jld2" eval_set
end

function save_data_special(num_start_states=500, seed=1)
    rets = gen_dataset_special(num_start_states, seed)
    eval_set = Dict(
        "ests" => rets[1],
        "actions" => rets[3],
        "states" => rets[2]
    )
    @save "./src/data/TwoDGridWorldSet_dpi.jld2" eval_set
end


end
