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
    feature_projector::Any

end

function get_weights(learner::SRLearner)
    return [learner.ψ, learner.r_w]
end

function SRLearner{F}(lu, feature_size, num_demons, num_actions, num_tasks, fp) where {F<:Number}
    SRLearner(zeros(F, num_demons-num_tasks, feature_size * num_actions),
              zeros(F, num_tasks, size(fp) * num_actions),
              lu,
              num_demons,
              num_actions,
              feature_size,
              num_tasks, fp)
end

SRLearner(lu, feature_size, num_demons, num_actions, num_tasks, fp) =
    SRLearner{Float64}(lu, feature_size, num_demons, num_actions, num_tasks, fp)

update(l::SRLearner) = l.update

Base.size(learner::SRLearner) = learner.num_demons

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
