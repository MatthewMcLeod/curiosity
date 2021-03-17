

@kwdef mutable struct TB{O} <: Learner
    lambda::Float64
    opt::O
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

get_action_inds(action, num_actions, num_gvfs) = [action + (i-1)*num_actions for i in 1:num_gvfs]

function update!(lu::TB,
                 learner::QLearner{Matrix{<:AbstractFloat}},
                 demons,
                 obs,
                 next_obs,
                 state,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 behaviour_pi_func)

    weights = learner.model
    λ = lu.lambda

    C, next_discounts, _ = get(demons, obs, action, next_obs, next_action)
    target_pis = get_demon_pis(demons, learner.num_actions, state, obs)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)

    discounts = get!(lu.prev_discounts, learner, zero(next_discounts))::typeof(next_discounts)
    e = get!(lu.e, weights, zero(weights))::typeof(weights)

    # Update eligibility trace
    #Broadcast the policy and pseudotermination of each demon across the actions
    e .*= λ * repeat(discounts, inner = learner.num_actions) .* repeat(target_pis[:,action], inner = learner.num_actions)

    # the eligibility trace is for state-action so need to find the exact state action pair per demon and not just the state
    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds
    e[inds,:] .+= state'

    pred = learner(next_state)
    Qs = reshape(pred, (learner.num_actions, learner.num_demons))'

    # Target Pi is num_demons  x num_actions
    backup_est_per_demon = vec(sum((Qs .* next_target_pis ), dims = 2))
    target = C + next_discounts .* backup_est_per_demon

     # TD error per demon is the td error on Q
    td_err = target - (weights * state)[inds]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    # update discounts
    discounts .= next_discounts

    if learner.opt isa Auto
        next_state_action_row_ind = get_action_inds(next_action, learner.num_actions, learner.num_demons)
        state_discount = zero(e)
        state_discount[state_action_row_ind,:] .+= state'
        state_discount[next_state_action_row_ind,:] .-= next_discounts * next_state'
        abs_phi = abs.(e)
        update!(learner.opt, weights, e, td_err_across_demons, abs_phi .* max.(state_discount, abs_phi))
    else
        Flux.Optimise.update!(learner.opt, weights,  -(e .* td_err_across_demons))
    end
end

function zero_eligibility_traces!(learner::TB)
    # learner.e .= 0
    for (k, v) ∈ learner.e
        v .= 0
    end
end

# function get_weights(learner::TB, weights)
#     return weights
# end

# function predict(learner::TB, agent, weights::Array{Float64,2}, obs, action)
#     state = agent.state_constructor(obs)
#     preds = weights * state
#     inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
#     return preds[inds]
# end
