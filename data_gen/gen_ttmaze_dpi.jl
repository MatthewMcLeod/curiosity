module GenTabularTMazeDpi

using Curiosity
using Curiosity: DPI, HordeDPI
using Curiosity: OpenWorld
using MinimalRLCore
const TTMU = Curiosity.TabularTMazeUtils
include("../experiments/tabular_tmaze.jl")
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

function gen_dpi(num_states=100000, seed=1; exploring_starts=false, h_kwargs...)
    Random.seed!(seed)
    
    parsed = TabularTMazeExperiment.default_args()
    parsed["exploring_starts"] = exploring_starts
    parsed["demon_learner"] = "Q"

    dpis = DPI[]

    state_filter_func =  Curiosity.FeatureSubset(identity, 1)
    
    for gvf_i ∈ 1:4
        cumulant_schedule = TTMU.get_cumulant_schedule(parsed)
        exploring_starts = parsed["exploring_starts"]
        env = TabularTMaze(exploring_starts, cumulant_schedule)

        ss = Float64[]
        as = Int[]
        policy = GVFParamFuncs.FunctionalPolicy((;kwargs...) -> TTMU.demon_target_policy(gvf_i;kwargs...))
        
        while length(ss) < num_states
            s = MinimalRLCore.start!(env)
            push!(ss, state_filter_func(s))
            a = sample(policy, s, 1:4)

            for t ∈ 1:rand(1:15)
                s, r, term = MinimalRLCore.step!(env, a)
                a = sample(policy, s, 1:4)
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

function save_dpi(num_states, seed; exploring_starts=false, h_kwargs...)
    hdpi = gen_dpi(num_states, seed; exploring_starts=exploring_starts, h_kwargs...)
    @save "./src/data/dpi/ttmaze_exploring_starts_$(exploring_starts).jld2" hdpi
end

end