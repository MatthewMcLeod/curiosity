
struct FeatureSubset{T, R} <: FeatureCreator
    fc::T
    subset::R
end

Base.size(fc::FeatureSubset) = Base.size(fc.fc)
Base.length(fc::FeatureSubset) = Base.size(fc.fc)

(fc::FeatureSubset)(s; kwargs...) =
    fc.fc(s[fc.subset]; kwargs...)

(fc::FeatureSubset)(s,a,s_tp1) = fc.fc(s[fc.subset],a,s_tp1[fc.subset])

struct Sparsify{FC}
    fc::FC
end

function (fc::Sparsify)(s; kwargs...)
    ints = fc.fc(s; kwargs...)
    s = spzeros(size(fc))
    if ints isa AbstractArray
        s[ints] .= 1
    else
        s[ints] = 1
    end
    return s
end

function get_active_action_state_vector(state::SparseVector, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end

mutable struct ValueFeatureProjector <: FeatureCreator
    func::Function
    pf_length::Int
end

function project_features(FP::ValueFeatureProjector, state)
    return FP.func(state, action)
end
(FP::ValueFeatureProjector)(state,action) = project_features(FP, state, action)

Base.size(VFP::ValueFeatureProjector) = VFP.pf_length

mutable struct ActionValueFeatureProjector{T} <: FeatureCreator
    func::T
    num_actions::Int
end

function project_features(FP::ActionValueFeatureProjector,state,action,state_tp1)
    state_action = spzeros(size(FP.func) * FP.num_actions)
    state = FP.func(state,action,state_tp1)
    state_length = size(FP.func)
    state_action[(action-1)*state_length + 1: action*state_length] = state
    return state_action
end

(FP::ActionValueFeatureProjector)(state,action,next_state) = project_features(FP, state, action, next_state)

Base.length(AVFP::ActionValueFeatureProjector) = size(AVFP.func) * AVFP.num_actions
Base.size(AVFP::ActionValueFeatureProjector) = size(AVFP.func) * AVFP.num_actions


struct FeatureProjector{FC} <: FeatureCreator
    fc::FC
    next_obs::Bool
end

Base.size(fc::FeatureProjector) = Base.size(fc.fc)
Base.length(fc::FeatureProjector) = Base.size(fc.fc)

(fc::FeatureProjector)(s, a, s_prime; kwargs...) = if fc.next_obs
    fc.fc(s_prime; kwargs...)
else
    fc.fc(s; kwargs...)
end
