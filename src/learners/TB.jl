

mutable struct TB{O} <: Learner
    lambda::Float64
    opt::O
    # e::Array{Float32, 2}
    e::IdDict
    num_demons::Int
    num_actions::Int
    feature_size::Int
    # alpha::Float64
    function TB(lambda, opt, feature_size, num_demons, num_actions)
        # new(lambda, zeros(num_actions * num_demons, feature_size), num_demons, num_actions, alpha)
        # @show (lambda, opt, IdDict(), num_demons, num_actions)
        new{typeof(opt)}(lambda, opt, IdDict(), num_demons, num_actions, feature_size)
    end
end
# Base.size(learner::TB) = size(learner.e)
Base.size(learner::TB) = (learner.num_demons * learner.num_actions, learner.feature_size)

get_prediction(w::Matrix, s) = w*s

get_action_inds(action, num_actions, num_gvfs) = [action + (i-1)*num_actions for i in 1:num_gvfs]

function update!(learner::TB,
                 weights,
                 C,
                 state,
                 action,
                 target_pis,
                 discounts,
                 next_state,
                 next_action,
                 next_target_pis,
                 next_discounts)
    # Update eligibility trace
    #Broadcast the policy and pseudotermination of each demon across the actions
    e = get!(learner.e, weights, zero(weights))::typeof(weights)

    e .*= learner.lambda * repeat(discounts, inner = learner.num_actions) .* repeat(target_pis[:,action], inner = learner.num_actions)

    # the eligibility trace is for state-action so need to find the exact state action pair per demon and not just the state
    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds
    e[inds,:] .+= state'

    pred = get_prediction(weights, next_state)
    Qs = reshape(pred, (learner.num_actions, learner.num_demons))'

    # Target Pi is num_demons  x num_actions
    backup_est_per_demon = vec(sum((Qs .* next_target_pis ), dims = 2))
    target = C + next_discounts .* backup_est_per_demon

     # TD error per demon is the td error on Q
    td_err = target - (weights * state)[inds]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    # How to efficiently apply gradients back into weights? Should we move linear regression to Flux/Autograd?
    # TODO: Seperate optimizer and learning algo
    # weights .= weights + learner.alpha * (e .* td_err_across_demons)
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

function get_weights(learner::TB, weights)
    return weights
end

function predict(learner::TB, agent, weights::Array{Float64,2}, obs, action)
    state = agent.state_constructor(obs)
    preds = weights * state
    inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
    return preds[inds]
end
