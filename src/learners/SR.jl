using SparseArrays
using LinearAlgebra

mutable struct SR{F<:Number} <: Learner
    pred_weights::Matrix{F}
    ψ::Matrix{F}
    r_w::Matrix{F}
    
    num_demons::Int
    num_actions::Int
    
    feature_size::Int
    
    num_tasks::Int
    
end

function SR(feature_size, num_demons, num_actions, num_tasks)
    SR(zeros(num_tasks, feature_size * num_actions),
       zeros(num_demons, feature_size * num_actions),
       ones(num_demons, feature_size * num_actions),
       num_demons, num_actions, 
       feature_size,
       num_tasks)
end

# TODO: Does it makes sense to have the size be the size of the model parameters? Not really.... Should be size of the output
Base.size(learner::SR) = size(learner.ψ) #(learner.num_demons, learner.feature_size * learner.num_actions)

function get_active_action_state_vector(state::SparseVector, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end

predict_sf(learner::SR, ϕ::SparseVector) = learner.ψ[:, active_state_action.nzind] * active_state_action.nzval

function predict(learner::SR, ϕ::SparseVector, action)#agent, weights::Array{Float64,2}, obs, action)
    # state = agent.state_constructor(obs)
    active_state_action = get_active_action_state_vector(ϕ, action, length(ϕ), learner.num_actions)

    SF = predict_SF(learner, ϕ)

    #Column is SF per task
    reshaped_SF = reshape(SF, length(active_state_action), learner.num_tasks)
    
    Q = learner.pred_weights * reshaped_SF
    return Q[diagind(Q)]
end

(l::SR)(ϕ, a) = predict(l, ϕ, a)
(l::SR)(ϕ) = [predict(l, ϕ, a) for a ∈ 1:num_actions]
