

Base.@kwdef mutable struct TBTrueOnline{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    trace::T = AccumulatingTraces()
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
    prev_weights::IdDict = IdDict()
end

get_action_inds(action, num_actions, num_gvfs) = [action + (i-1)*num_actions for i in 1:num_gvfs]

function get_demon_parameters(lu::TBTrueOnline, learner, demons, obs, state, action, next_obs, next_state, next_action, env_reward)
    # C, next_discounts, _ = get(demons, obs, action, next_obs, next_action)
    C, next_discounts, _ = get(demons; state_t = obs, action_t = action, state_tp1 = next_obs, action_tp1 = next_action, reward = env_reward)
    target_pis = get_demon_pis(demons, learner.num_actions, state, obs)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)
    C, next_discounts, target_pis, next_target_pis
end


function update!(lu::TBTrueOnline,
                 learner::QLearner{M, LU},
                 demons,
                 obs,
                 next_obs,
                 state,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 behaviour_pi_func,
                 env_reward) where {M<:AbstractMatrix, LU<:TBTrueOnline}

    weights = learner.model
    λ = lu.lambda

    C, next_discounts, target_pis, next_target_pis =
        get_demon_parameters(lu,
                             learner,
                             demons,
                             obs,
                             state,
                             action,
                             next_obs,
                             next_state,
                             next_action,
                             env_reward)

    discounts = get!(()->zero(next_discounts), lu.prev_discounts, learner)::typeof(next_discounts)
    e = get!(()->zero(weights), lu.e, weights)::typeof(weights)
    prev_weights = get!(() -> zero(e), lu.prev_weights, weights)::typeof(weights)

    # Update eligibility trace
    #Broadcast the policy and pseudotermination of each demon across the actions
    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds

    #NOTE True Online Trace
    π_A_t = repeat(target_pis[:,action], inner = learner.num_actions)
    γ_t = repeat(discounts, inner = learner.num_actions)
    term1 = π_A_t .* γ_t .* λ
    α = lu.opt.eta

    active_state_action = get_active_action_state_vector(state, action, length(state), learner.num_actions)
    ϕ_S_a = repeat(reshape(active_state_action, learner.num_actions, :), learner.num_demons, 1)
    term2 = α*(1 .- sum((term1 .* ϕ_S_a) .* e, dims = 2)) .* ϕ_S_a

    e .= term1 .* e + term2

    pred = learner(next_state)
    Qs = reshape(pred, (learner.num_actions, learner.num_demons))'

    # Target Pi is num_demons  x num_actions
    backup_est_per_demon = vec(sum((Qs .* next_target_pis ), dims = 2))
    target = C + next_discounts .* backup_est_per_demon

     # TD error per demon is the td error on Q
    td_err = target - (prev_weights * state)[inds]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    # update discounts
    discounts .= next_discounts

    # if lu.opt isa Auto
    #     next_state_action_row_ind = get_action_inds(next_action, learner.num_actions, learner.num_demons)
    #     state_discount = zero(e)
    #     state_discount[state_action_row_ind,:] .+= state'
    #     state_discount[next_state_action_row_ind,:] .-= next_discounts * next_state'
    #     abs_phi = abs.(e)
    #     update!(lu.opt, weights, e, td_err_across_demons, abs_phi .* max.(state_discount, abs_phi), learner.num_demons, learner.num_actions)
    # else
    #     Flux.Optimise.update!(lu.opt, weights,  -(e .* td_err_across_demons))
    # end
    # weights .= weights + td_err_across_demons * e + α*(ϕ * weights - ϕ' * weights) * ϕ

    #NOTE: I don't see how this kind of update fits into the flux framework...
    #TODO: Comeback and fix this. Priority currently on reproducing experiments
    if (lu.opt isa Descent)
        new_weights = weights + td_err_across_demons .* e + α .* (sum((ϕ_S_a .* prev_weights), dims = 2) .- sum((ϕ_S_a .* weights), dims = 2) ) .* ϕ_S_a
    else
        throw(DomainError("Only SGD is supported for True Online. TODO: extend to using flux"))
    end

    prev_weights .= deepcopy(weights)
    weights .= new_weights

end

function zero_eligibility_traces!(lu::TBTrueOnline)
    # learner.e .= 0
    for (k, v) ∈ lu.e
        if eltype(v) <: Integer
            lu.e[k] = Int[]
        else
            v .= 0
        end
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
