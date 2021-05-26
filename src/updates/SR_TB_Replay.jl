


function update_rew_part!(lu::TB,
                          learner::SRLearner,
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

    discounts = get!(()->zero(next_discounts), lu.prev_discounts, learner.r_w)::typeof(next_discounts)
    e_w_nz = get!(()->Int[], lu.e, learner)::Array{Int, 1}
    e_w = get!(()->zero(w), lu.e, w)::typeof(w)
    λ = lu.lambda

    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action, length(state), learner.num_actions)

    projected_state = learner.feature_projector(obs, action, next_obs)

    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]

    # Update Traces: See update_utils.jl
    update_trace!(lu.trace, e_w, projected_state, λ, reward_discounts, reward_target_pis[:, action])
    # e_nz = e_nz ∪ active_state_action.nzind
    # e_w_nz = e_w_nz ∪ projected_state_action.nzind
    if λ == 0.0
        e_w_nz = projected_state.nzind
    else
        e_w_nz = e_w_nz ∪ projected_state.nzind
        lu.e[learner] = e_w_nz
    end
    
    pred_err = reward_C - w * projected_state

    # This should always be true as this is immediate next step prediction which is equivalent to having discounts of 0 for all states
    @assert sum(reward_discounts) == 0

    # TD err is applied across rows
    opt = if lu.opt isa Tuple
        lu.opt[1]
    else
        lu.opt
    end
    if opt isa Auto

        state_discount_r = projected_state'
        abs_ϕ_w = if λ == 0.0
            abs.(repeat(projected_state, outer=(1, length(pred_err)))')
        else
            abs.(e_w)
        end
        z_r = abs_ϕ_w .* max.(abs_ϕ_w, state_discount_r)
        update!(opt, w, e_w, pred_err, z_r, learner.num_tasks, 1)
        
    elseif opt isa Flux.Descent
        α = opt.eta
        if λ == 0.0
            w .= w .+ α * pred_err * projected_state'
        else
            w[:, e_w_nz] .+= (α  * pred_err) .* e_w[:, e_w_nz]
        end
    else
        update!(opt, w, - e_w .* pred_err)
    end
    discounts .= next_discounts
end

function update_sr_part!(lu::TB,
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
                 env_reward; lambda = false)

    ψ = learner.ψ

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

    discounts = get!(()->zero(next_discounts), lu.prev_discounts, learner.ψ)::typeof(next_discounts)
    e_nz = get!(()->Int[], lu.e, learner)::Array{Int, 1}
    e_ψ = get!(()->zero(ψ), lu.e, ψ)::typeof(ψ)
    λ = lu.lambda * lambda

    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action, length(state), learner.num_actions)

    projected_state = learner.feature_projector(obs, action, next_obs)

    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]

    # Update Traces: See update_utils.jl
    update_trace!(lu.trace, e_ψ, active_state_action, λ, SF_discounts, SF_target_pis[:, action])

    if λ == 0.0
        e_nz = active_state_action.nzind
    else
        e_nz = e_nz ∪ active_state_action.nzind
        lu.e[learner] = e_nz
    end


    pred = ψ * next_active_state_action
    reward_feature_backup = zeros(length(SF_C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += SF_next_target_pis[:,a] .* (ψ * next_possible_active_state_action)
    end

    target = SF_C + SF_next_discounts .* reward_feature_backup
    td_err = target - ψ * active_state_action

    # This should always be true as this is immediate next step prediction which is equivalent to having discounts of 0 for all states
    @assert sum(reward_discounts) == 0
    # TD err is applied across rows

    opt = if lu.opt isa Tuple
        lu.opt[2]
    else
        lu.opt
    end
    if opt isa Auto
        # next_state_action_row_ind = get_action_inds(next_action, learner.num_actions, learner.num_demons)
        state_discount = -SF_next_discounts * next_active_state_action'
        state_discount .+= active_state_action'
        abs_ϕ_ψ = if λ == 0.0
            abs.(repeat(active_state_action, outer=(1, length(td_err)))')
        else
            abs.(e_ψ)
        end
        z = abs_ϕ_ψ .* max.(abs_ϕ_ψ, state_discount)
        update!(opt, ψ, e_ψ, td_err, z,  learner.num_demons - learner.num_tasks, 1)

    elseif opt isa Flux.Descent
        α = opt.eta
        if λ == 0.0
            ψ[:, active_state_action.nzind] .+= (α  * td_err) * active_state_action.nzval'
        else
            ψ[:, e_nz] .+= (α  * td_err) .* e_ψ[:, e_nz]
        end
    else
        update!(opt, ψ, - e_ψ .* td_err)
    end
    discounts .= next_discounts
end
