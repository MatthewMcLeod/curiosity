using SparseArrays

mutable struct SF <: Learner
    lambda::Float64
    e::Array{Float64,2}
    num_demons::Int
    num_actions::Int
    alpha::Float64
    weight_cutoff::Int
    immediate_reward_predictor::Array{Float64,2}
    feature_size::Int
    function SF(lambda, feature_size, num_demons, num_actions, alpha)
        new(lambda, zeros(num_demons, feature_size), num_demons, num_actions, alpha, num_demons, ones(num_demons,feature_size * num_actions), feature_size)
    end
end

Base.size(learner::SF) = (learner.num_demons, learner.feature_size * learner.num_actions)

function get_active_action_state_vector(state::AbstractArray, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    ind_adjustment = feature_size * (action - 1)
    active_state_action = sparsevec(state.nzind .+ ind_adjustment, state.nzval, vec_length)
    return active_state_action
end

function update!(learner::SF, weights_full, C, state, action, target_pis, discounts, next_state, next_action, next_target_pis, next_discounts)
    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action,length(state), learner.num_actions)
    weights = view(weights_full,1:learner.weight_cutoff,:)
    immediate_reward_estimator = view(learner.immediate_reward_predictor, 1:learner.weight_cutoff,:)

    pred = weights * next_active_state_action
    reward_feature_backup = zeros(length(C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += next_target_pis[:,a] .* (weights * next_possible_active_state_action)
    end

    target = C + next_discounts .* reward_feature_backup

    td_err = target - weights * active_state_action
    #td_err is (336x1)
    #TD err is applied across rows
    weights .= weights .+ learner.alpha * td_err * active_state_action'
end

function zero_eligibility_traces!(learner::SF)
    learner.e .= 0
end

function get_weights(learner::SF, weights)
    return weights[1:learner.weight_cutoff,:]
end

function predict_SF(learner::SF, agent, weights_full::Array{Float64,2}, obs, action)
    state = agent.state_constructor(obs)
    active_state_action = get_active_action_state_vector(state, action,length(state), learner.num_actions)

    # println("obs: ", obs)
    weights = weights_full[1:learner.weight_cutoff,:]
    #
    # println("W:  ", size(weights), "  S:", println(size(state)))
    # println("STATE: ", state)
    # println("LENGTH: ", length(state))

    preds = weights * active_state_action
    return preds
    # inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
    # return preds[inds]
end
function predict(learner::SF, agent, weights_full::Array{Float64,2}, obs, action)
    SF = predict_SF(learner,agent,weights_full, obs, action)
    return SF
    #Reshape it into the SF per demon
    # reshaped_SF = reshape(SF,length(active_state_action), Int(length(SF)/length(active_state_action)))'
    #Repeat the predicted feature vector per set of GVFs
    # full_SF = repeat(reshaped_SF,inner=(length(active_state_action),1))

end
