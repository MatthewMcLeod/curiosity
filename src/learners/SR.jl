using SparseArrays
using LinearAlgebra

mutable struct SRLearner{F<:Number, LU<:LearningUpdate} <: Learner
    ψ::Matrix{F}
    r_w::Matrix{F}
    update::LU
    
    num_demons::Int
    num_actions::Int
    
    feature_size::Int
    
    num_tasks::Int
    
end

function SRLearner{F}(lu, feature_size, num_demons, num_actions, num_tasks) where {F<:Number}
    SRLearner(zeros(F, num_demons-num_tasks, feature_size * num_actions),
              zeros(F, num_tasks, feature_size * num_actions),
              lu,
              num_demons,
              num_actions, 
              feature_size,
              num_tasks)
end

SRLearner(lu, feature_size, num_demons, num_actions, num_tasks) =
    SRLearner{Float64}(lu, feature_size, num_demons, num_actions, num_tasks)

update(l::SRLearner) = l.update

Base.size(learner::SRLearner) = learner.num_demons

function get_active_action_state_vector(state::SparseVector, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end

function predict_SF(learner::SRLearner, ϕ::SparseVector, action)
    active_state_action = get_active_action_state_vector(ϕ, action, length(ϕ), learner.num_actions)
    learner.ψ[:, active_state_action.nzind] * active_state_action.nzval
end

function predict(learner::SRLearner, ϕ::SparseVector, action)

    SF = predict_SF(learner, ϕ, action)

    #Column is SF per task
    reshaped_SF = reshape(SF, :, learner.num_tasks)
    
    Q = learner.r_w * reshaped_SF
    return Q[diagind(Q)]
end

(l::SRLearner)(ϕ, a) = predict(l, ϕ, a)
(l::SRLearner)(ϕ) = [predict(l, ϕ, a) for a ∈ 1:num_actions]
