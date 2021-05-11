module GenOneDTMazeDpi

using Curiosity
using Curiosity: DPI, HordeDPI
using Curiosity: OpenWorld
using MinimalRLCore
include("../experiments/1d-tmaze.jl")
const ODTMU = Curiosity.OneDTMazeUtils
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

function gen_dpi(num_states=100000, seed=1; exploring_starts="beg", h_kwargs...)
    Random.seed!(seed)
    
    parsed = OneDTmazeExperiment.default_args()
    parsed["exploring_starts"] = exploring_starts
    parsed["demon_learner"] = "Q"

    dpis = DPI[]

    state_filter_func =  Curiosity.FeatureSubset(identity, 1:2)
    
    for gvf_i ∈ 1:4
        cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)
        env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"])

        ss = Vector{Float64}[]
        as = Int[]
        policy = ODTMU.GoalPolicy(gvf_i)
        
        while length(ss) < num_states
            s = MinimalRLCore.start!(env)
            push!(ss, state_filter_func(s))

            a = policy(s)
            term = false
            while term == false
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

function save_dpi(num_states, seed; exploring_starts="beg", h_kwargs...)
    hdpi = gen_dpi(num_states, seed; exploring_starts=exploring_starts, h_kwargs...)
    @save "./src/data/dpi/oned_tmaze_$(exploring_starts).jld2" hdpi
end

end