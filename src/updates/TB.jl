

Base.@kwdef mutable struct TB{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    trace::T = AccumulatingTraces()
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

get_action_inds(action, num_actions, num_gvfs) = [action + (i-1)*num_actions for i in 1:num_gvfs]

function get_demon_parameters(lu::TB, learner, demons, obs, state, action, next_obs, next_state, next_action, env_reward)
    # C, next_discounts, _ = get(demons, obs, action, next_obs, next_action)
    C, next_discounts, _ = get(demons; state_t = obs, action_t = action, state_tp1 = next_obs, action_tp1 = next_action, reward = env_reward)
    target_pis = get_demon_pis(demons, learner.num_actions, state, obs)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)
    C, next_discounts, target_pis, next_target_pis
end


function update!(lu::TB,
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
                 env_reward) where {M<:AbstractMatrix, LU<:TB}

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

    # Update eligibility trace
    #Broadcast the policy and pseudotermination of each demon across the actions
    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds

    update_trace!(lu.trace,
                  e,
                  state,
                  λ,
                  repeat(discounts, inner = learner.num_actions),
                  repeat(target_pis[:,action], inner = learner.num_actions),
                  inds)


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

    if lu.opt isa Auto
        next_state_action_row_ind = get_action_inds(next_action, learner.num_actions, learner.num_demons)
        state_discount = zero(e)
        state_discount[state_action_row_ind,:] .+= state'
        state_discount[next_state_action_row_ind,:] .-= next_discounts * next_state'
        abs_phi = abs.(e)
        update!(lu.opt, weights, e, td_err_across_demons, abs_phi .* max.(state_discount, abs_phi), learner.num_demons, learner.num_actions)
    else
        Flux.Optimise.update!(lu.opt, weights,  -(e .* td_err_across_demons))
    end
end

function update!(lu::TB,
                 learner::Union{SRLearner,GPI},
                 demons,
                 obs,
                 next_obs,
                 state::SparseVector,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 behaviour_pi_func,
                 env_reward)

    ψ = learner.ψ
    w = learner.r_w

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
    e_nz = get!(()->Int[], lu.e, learner)::Array{Int, 1}
    e_w_nz = get!(()->Int[], lu.e, learner)::Array{Int, 1}
    e_ψ = get!(()->zero(ψ), lu.e, ψ)::typeof(ψ)
    e_w = get!(()->zero(w), lu.e, w)::typeof(w)
    λ = lu.lambda

    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action, length(state), learner.num_actions)

    projected_state = learner.feature_projector(obs)
    projected_state_action = get_active_action_state_vector(projected_state, action, length(learner.feature_projector), learner.num_actions)
    # projected_next_state_next_action = learner.feature_projector(next_state,next_action)

    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]


    # Update Traces: See update_utils.jl
    if λ !== 0.0
        update_trace!(lu.trace, e_ψ, active_state_action, λ, SF_discounts, SF_target_pis[:, action])
        # update_trace!(lu.trace, e_w, projected_state_action, λ, reward_discounts, reward_target_pis[:, action])
        #Reward learning is a supervised learning problem so discounts = 0
        update_trace!(lu.trace, e_w, projected_state_action, λ, zeros(size(reward_discounts)), reward_target_pis[:, action])
        e_nz = e_nz ∪ active_state_action.nzind
        e_w_nz = e_w_nz ∪ projected_state_action.nzind
    end

    pred = ψ * next_active_state_action
    reward_feature_backup = zeros(length(SF_C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += SF_next_target_pis[:,a] .* (ψ * next_possible_active_state_action)
    end

    target = SF_C + SF_next_discounts .* reward_feature_backup
    td_err = target - ψ * active_state_action

    pred_err = reward_C - w * projected_state_action
    #td_err is (336x1)
    # TD err is applied across rows

    if lu.opt isa Auto
        # next_state_action_row_ind = get_action_inds(next_action, learner.num_actions, learner.num_demons)
        state_discount = -SF_next_discounts * next_active_state_action'
        state_discount .+= active_state_action'
        abs_ϕ_ψ = if λ == 0.0
            abs.(repeat(active_state_action, outer=(1, length(td_err)))')
        else
            abs.(e_ψ)
        end
        z = abs_ϕ_ψ .* max.(abs_ϕ_ψ, state_discount)
        update!(lu.opt, ψ, e_ψ, td_err, z,  learner.num_demons - learner.num_tasks, 1)

        state_discount_r = -reward_next_discounts * next_active_state_action'
        state_discount_r .+= active_state_action'
        abs_ϕ_w = if λ == 0.0
            abs.(repeat(active_state_action, outer=(1, length(pred_err)))')
        else
            abs.(e_w)
        end
        z_r = abs_ϕ_w .* max.(abs_ϕ_w, state_discount_r)
        update!(lu.opt, w, e_w, pred_err, z_r, learner.num_tasks, 1)
        # throw("SR + TB + Auto not implemented")
    elseif lu.opt isa Flux.Descent
        α = lu.opt.eta
        if λ == 0.0
            ψ[:, active_state_action.nzind] .+= (α  * td_err) * active_state_action.nzval'
            w .= w .+ α * pred_err * active_state_action'
        else
            ψ[:, e_nz] .+= (α  * td_err) .* e_ψ[:, e_nz]
            w[:, e_w_nz] .+= (α  * pred_err) .* e_w[:, e_w_nz]
        end
    else
        update!(lu.opt, ψ, -td_err .* e_ψ)
        update!(lu.opt, w, -pred_err * e_w)
    end
    discounts .= next_discounts
end


function zero_eligibility_traces!(lu::TB)
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


function update!(lu::TB,
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

         e .= γ_t[gvf]*λ*π_t[gvf] * e + x_t
         b .+= (c*e - b)/(t+1)

         u = sum(π_tp1[gvf, a] .* get_active_action_state_vector(ϕ_tp1, a, fs, na) for a ∈ 1:na)
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
