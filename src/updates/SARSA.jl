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

function zero_all_nonfinites!(itm)
    if eltype(itm) <: Number
        itm[findall(.!isfinite.(itm))] .= 0.0

    elseif eltype(itm) <: Vector
        [zero_all_nonfinites(lst) for lst in itm]
    else
        throw(ArgumentError("Not Implemented"))
    end
end

function update!(lu::SARSA,
                 learner::LSTDLearner,
                 demons,
                 obs,
                 next_obs,
                 state,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 behaviour_pi_func,
                 env_reward)

     C, γ_tp1, π_t, π_tp1 =
         get_demon_parameters(learner,
                              demons,
                              obs,
                              state,
                              action,
                              next_obs,
                              next_state,
                              next_action)

     na = learner.num_actions
     fs = learner.feature_size
     λ = lu.lambda
     t = learner.t
     μ_t = behaviour_pi_func(state, obs)[action]
     μ_tp1 = behaviour_pi_func(next_state, next_obs)

     ρ_t = π_t ./ μ_t
     ρ_tp1 = π_tp1 ./ μ_tp1'

     zero_all_nonfinites!(ρ_t)
     zero_all_nonfinites!(ρ_tp1)


     γ_t = get!(()->zero(γ_tp1), lu.prev_discounts, learner)::typeof(γ_tp1)

     ϕ_t = state
     ϕ_tp1 = next_state

     x_t = get_active_action_state_vector(
         ϕ_t, action, learner.feature_size, learner.num_actions)
     x_tp1 = get_active_action_state_vector(
         ϕ_tp1, next_action, learner.feature_size, learner.num_actions)


    all_gvf_e = get!(()->zero.(learner.w_real), lu.e, learner.w_real)::typeof(learner.w_real)


     for gvf ∈ 1:learner.num_demons
         A_inv = learner.A_inv[gvf]
         e = all_gvf_e[gvf]
         b = learner.b[gvf]
         c = C[gvf]

         # NOTE TD Update
         e .= γ_t[gvf]*λ*ρ_t[gvf] * e + x_t
         #NOTE TB Update
         # e .= γ_t[gvf]*λ*π_t[gvf] * e + x_t
         b .+= (c*e - b)/(t+1)

         # NOTE TD UPDATE
         u = sum(ρ_tp1[gvf, a] .* get_active_action_state_vector(ϕ_tp1, a, fs, na) for a ∈ 1:na)
         # NOTE TB Update
         # u = sum(π_tp1[gvf, a] .* get_active_action_state_vector(ϕ_tp1, a, fs, na) for a ∈ 1:na)
         v = transpose(transpose(x_t - γ_tp1[gvf]*u) * A_inv)

         if t > 0
             scale = (t+1)/t
             vz = dot(v, e)
             Ainv_zvt = (A_inv * e) * transpose(v)
             A_inv .= scale * (A_inv - Ainv_zvt./(t + vz))
         else
             Aev = A_inv*(e*v')
             ve = dot(v, e)
             A_inv .-= Aev/(1 + ve)
         end
         learner.w_real[gvf] .= A_inv * b
     end
     γ_t .= γ_tp1
     learner.t += 1
end



function zero_eligibility_traces!(lu::SARSA)
    for (k, v) ∈ lu.e
        if eltype(v) <: Integer
            lu.e[k] = Int[]
        elseif eltype(v) <: Vector #Used for LSTD
            lu.e[k] = zero.(v)
        else
            v .= 0
        end
    end
end
