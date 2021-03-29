Base.@kwdef mutable struct SARSA{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    trace::T
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

function update!(lu::SARSA,
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
                 reward) where {M<:AbstractMatrix, LU<:SARSA}

     weights = learner.model
     λ = lu.lambda
     e = get!(()->zero(weights), lu.e, weights)::typeof(weights)

     C, next_discounts, target_pis, next_target_pis = get_demon_parameters(lu, learner, demons, obs, state, action, next_obs, next_state, next_action, reward)
     b_πs = behaviour_pi_func(state, obs)
     discounts = get!(()->zero(next_discounts), lu.prev_discounts, learner)::typeof(next_discounts)

     #NOTE: Off policy SARSA(λ) I think is the same as EASRSA except the backup is sample and not expected.
     ρ = if b_πs[action] == 0
         zeros(length(target_pis[:,action]))
     else
         target_pis[:,action] ./ b_πs[action]
     end

     inds = get_action_inds(action, learner.num_actions, learner.num_demons)
     state_action_row_ind = inds

     update_trace!(lu.trace,
                   e,
                   state,
                   λ,
                   repeat(discounts, inner = learner.num_actions),
                   repeat(ρ, inner = learner.num_actions),
                   inds)

     next_preds = learner(next_state, next_action)
     pred = learner(state, action)

     td_err = C .+ next_discounts .* next_preds .- pred
     td_err_across_demons = repeat(vec(td_err), inner=learner.num_actions)

     if lu.opt isa Auto
         next_state_action_row_ind = get_action_inds(next_action, learner.num_actions, learner.num_demons)
         state_discount = zero(e)
         state_discount[state_action_row_ind,:] .+= state'
         state_discount[next_state_action_row_ind,:] .-= next_discounts * next_state'
         abs_phi = abs.(e)
         update!(lu.opt, weights, e, td_err_across_demons, abs_phi .* max.(state_discount, abs_phi))
     else
         Flux.Optimise.update!(lu.opt, weights,  -(e .* td_err_across_demons))
     end
     discounts .= next_discounts
end



function zero_eligibility_traces!(lu::SARSA)
    for (k, v) ∈ lu.e
        if eltype(v) <: Integer
            lu.e[k] = Int[]
        else
            v .= 0
        end
    end
end
