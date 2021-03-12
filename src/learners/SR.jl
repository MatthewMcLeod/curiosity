using SparseArrays
using LinearAlgebra

mutable struct SR <: Learner
    lambda::Float64
    e::Array{Float64,2}
    num_demons::Int
    num_actions::Int
    alpha::Float64
    immediate_reward_predictor::Array{Float64,2}
    feature_size::Int
    num_tasks::Int
    function SR(lambda, feature_size, num_demons, num_actions, alpha, num_tasks)
        new(lambda, zeros(num_demons, feature_size), num_demons, num_actions, alpha, ones(num_demons,feature_size * num_actions), feature_size,num_tasks)
    end
end

Base.size(learner::SR) = (learner.num_demons, learner.feature_size * learner.num_actions)

function get_active_action_state_vector(state::AbstractArray, action, feature_size, num_actions)

    vec_length = feature_size * num_actions
    new_ind = (state.nzind .- 1) * num_actions .+ action
    active_state_action = sparsevec(new_ind, state.nzval, vec_length)
    return active_state_action
end

function update!(learner::SR, weights, C, state, action, target_pis, discounts, next_state, next_action, next_target_pis, next_discounts)
    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action,length(state), learner.num_actions)

    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]

    immediate_reward_estimator = view(weights, 1:learner.num_tasks,:)

    SF_estimator =  view(weights,learner.num_tasks+1:length(reward_C) + length(SF_C),:)

    pred = SF_estimator * next_active_state_action
    reward_feature_backup = zeros(length(SF_C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += SF_next_target_pis[:,a] .* (SF_estimator * next_possible_active_state_action)
    end

    target = SF_C + SF_next_discounts .* reward_feature_backup
    td_err = target - SF_estimator * active_state_action
    #td_err is (336x1)
    #TD err is applied across rows
    SF_estimator .= SF_estimator .+ learner.alpha * td_err * active_state_action'

    pred_err = reward_C - immediate_reward_estimator * active_state_action
    immediate_reward_estimator .= immediate_reward_estimator .+ learner.alpha * (pred_err) * active_state_action'
end

function zero_eligibility_traces!(learner::SR)
    learner.e .= 0
end

function get_weights(learner::SR, weights)
    @show size(weights)
    return weights
end

function predict_SF(learner::SR, agent, weights::Array{Float64,2}, obs, action)
    state = agent.state_constructor(obs)
    active_state_action = get_active_action_state_vector(state, action,length(state), learner.num_actions)
    preds = weights[learner.num_tasks+1:end,:] * active_state_action
    return preds
end
function predict(learner::SR, agent, weights::Array{Float64,2}, obs, action)
    state = agent.state_constructor(obs)
    active_state_action = get_active_action_state_vector(state, action, length(state), learner.num_actions)

    SF = predict_SF(learner,agent,weights, obs, action)

    #Column is SF per task
    reshaped_SF = reshape(SF,length(active_state_action),learner.num_tasks)
    prediction_weights = weights[1:learner.num_tasks,:]
    Q = prediction_weights * reshaped_SF
    return Q[diagind(Q)]
end
