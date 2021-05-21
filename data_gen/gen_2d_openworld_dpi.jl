using Curiosity
using Curiosity: DPI, HordeDPI
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

function gen_dpi(num_states=500, seed=1; start_type=:center, h_kwargs...)
    Random.seed!(seed)
    parsed = TwoDGridWorldExperiment.default_args()
    parsed["cumulant_schedule"] = "Constant"
    parsed["cumulant"] = 1.0
    parsed["demon_learner"] = "Q"

    dpis = DPI[]

    state_filter_func =  Curiosity.FeatureSubset(identity, 1:2)
    
    for gvf_i ∈ 1:4
        horde = Horde(
        [GVF(GVFParamFuncs.FeatureCumulant(gvf_i+2),
             TDGWU.GoalTermination(0.95),
             TDGWU.GoalPolicy(gvf_i, true))])
        
        cumulant_schedule = TDGWU.get_cumulant_schedule(parsed)
        
        env = OpenWorld(10, 10,
                        cumulant_schedule=cumulant_schedule,
                        start_type=start_type, normalized=true)
        
        ss = Vector{Float64}[]
        as = Int[]
        policy = TDGWU.GoalPolicy(gvf_i, true)
        
        while length(ss) < num_states
            s = MinimalRLCore.start!(env)
            push!(ss, state_filter_func(s))
            
            a = policy(s)
            for t ∈ 1:rand(1:15)
                s, r, term = MinimalRLCore.step!(env, a)
                a = policy(s)
                push!(ss, state_filter_func(s))
                if term
                    break
                end
            end
        end

         push!(dpis, DPI(ss, policy; h_kwargs...))
    end
        
    return HordeDPI(dpis, state_filter_func)
end

function save_dpi(num_states, seed; start_type=:center, h_kwargs...)
    hdpi = gen_dpi(num_states, seed; start_type=start_type, h_kwargs...)
    @save "./src/data/dpi/2dOpenWorld_$(start_type).jld2" hdpi
end
