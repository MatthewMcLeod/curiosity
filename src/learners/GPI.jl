using SparseArrays
using LinearAlgebra

mutable struct GPI{F<:Number, LU<:LearningUpdate} <: Learner
    ψ::Matrix{F}
    r_w::Matrix{F}
    update::LU

    num_demons::Int
    num_actions::Int

    feature_size::Int

    num_tasks::Int

end

function get_weights(learner::GPI)
    return vcat(learner.ψ, learner.r_w)
end

function GPI{F}(lu, feature_size, num_demons, num_actions, num_tasks) where {F<:Number}
    GPI(zeros(F, num_demons-num_tasks, feature_size * num_actions),
              zeros(F, num_tasks, feature_size * num_actions),
              lu,
              num_demons,
              num_actions,
              feature_size,
              num_tasks)
end

GPI(lu, feature_size, num_demons, num_actions, num_tasks) =
    GPI{Float64}(lu, feature_size, num_demons, num_actions, num_tasks)

update(l::GPI) = l.update

Base.size(learner::GPI) = learner.num_demons

function get_active_action_state_vector(state::SparseVector, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end

function predict_SF(learner::GPI, ϕ::SparseVector, action)
    active_state_action = get_active_action_state_vector(ϕ, action, length(ϕ), learner.num_actions)
    learner.ψ[:, active_state_action.nzind] * active_state_action.nzval
end

function predict(learner::GPI, ϕ::SparseVector, action)

    SF = predict_SF(learner, ϕ, action)

    #Column is SF per task
    # reshaped_SF = reshape(SF, :, learner.num_tasks)
    reshaped_SF = reshape(SF, length(learner.r_w), :)

    Q = learner.r_w * reshaped_SF
    # @show size(Q), Q
    # return Q[diagind(Q)]
    return maximum(Q)
end

(l::GPI)(ϕ, a) = predict(l, ϕ, a)
(l::GPI)(ϕ) = [predict(l, ϕ, a) for a ∈ 1:l.num_actions]
