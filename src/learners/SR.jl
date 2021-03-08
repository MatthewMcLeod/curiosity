using SparseArrays

mutable struct SR <: Learner
    lambda::Float64
    e::Array{Float64,2}
    num_demons::Int
    num_actions::Int
    alpha::Float64
    weight_cutoff::Int
    function SR(lambda, feature_size, num_demons, num_actions, alpha)
        new(lambda, zeros(num_demons, feature_size), num_demons, num_actions, alpha, num_demons)
    end
end

function get_active_action_state_vector(state::AbstractArray, action, feature_size, num_actions)
    vec_length = feature_size * num_actions
    ind_adjustment = feature_size * (action - 1)
    active_state_action = sparsevec(state.nzind .+ ind_adjustment, state.nzval, vec_length)
    return active_state_action
end


function update!(learner::SR, weights_full, C, state, action, target_pis, discounts, next_state, next_action, next_target_pis, next_discounts)
    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action,length(state), learner.num_actions)
    weights = view(weights_full,1:learner.weight_cutoff,:)

    # Values are expected
    # println("Nonzero C at: ", findall(!iszero, C))
    # println("For state: ", state, " and action: ", action, " Sum of C: ", sum(C))

    pred = weights * next_active_state_action
    target = C + next_discounts .* pred

    td_err = target - weights * active_state_action
    #td_err is (336x1)

    # println(size(td_err))
    # println(size(active_state_action))
    # println("Adjustment: ", sum(learner.alpha * td_err * active_state_action'))
    #TD is applied across rows
    weights .= weights .+ learner.alpha * td_err * active_state_action'
    # println("Sum of Weights: ", sum(weights))
    # println("Sum of Weights Full: ", sum(weights_full))



    # # Update eligibility trace
    #
    # learner.e .*= learner.lambda * discounts .* target_pis[:,action]
    #
    # # the eligibility trace is for state-action so need to find the exact state action pair per demon and not just the state
    # inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
    # learner.e[inds,:] .+= state'
    #
    # pred = weights * next_state
    # Qs = reshape(pred, (learner.num_actions, learner.num_demons))'
    # # Target Pi is num_demons  x num_actions
    # backup_est_per_demon = vec(sum((Qs .* next_target_pis ), dims = 2))
    # target = C + next_discounts .* backup_est_per_demon
    #  # TD error per demon is the td error on Q
    # td_err = target - (weights * state)[inds]
    # td_err_across_demons = repeat(td_err, inner=learner.num_actions)
    #
    # # How to efficiently apply gradients back into weights? Should we move linear regression to Flux/Autograd?
    # # TODO: Seperate optimizer and learning algo
    # weights .= weights + learner.alpha * (learner.e .* td_err_across_demons)
end

function zero_eligibility_traces!(learner::SR)
    learner.e .= 0
end

function get_weights(learner::SR, weights)
    return weights[1:learner.weight_cutoff,:]
end

function predict(learner::SR, agent, weights_full::Array{Float64,2}, obs, action)
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
