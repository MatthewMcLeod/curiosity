using SparseArrays

function row_order_reshape(A, reshape_dims)
    #I dont think Julia natively supports row based reshape
    # https://github.com/JuliaLang/julia/issues/20311
    return transpose(reshape(transpose(A), reshape_dims...))
end


function get_active_action_state_vector(state::SparseVector, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end

mutable struct ValueFeatureProjector <: AbstractFeatureProjector
    func::Function
    pf_length::Int
end

function project_features(FP::ValueFeatureProjector, state)
    return FP.func(state,action)
end
(FP::ValueFeatureProjector)(state,action) = project_features(FP, state, action)

Base.size(VFP::ValueFeatureProjector) = VFP.pf_length

mutable struct ActionValueFeatureProjector <: AbstractFeatureProjector
    func::Function
    pf_length::Int
end

function project_features(FP::ActionValueFeatureProjector,state)
    return FP.func(state)
end

# function project_state_action_features(FP::ActionValueFeatureProjector, state, action)
#     return FP.func(state, action)
# end

(FP::ActionValueFeatureProjector)(state) = project_features(FP, state)
# (FP::ActionValueFeatureProjector)(state, action) = project_features(FP, state, action)

Base.length(AVFP::ActionValueFeatureProjector) = AVFP.pf_length
