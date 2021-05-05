Base.@kwdef mutable struct ESARSA{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    # trace::T = ReplacingTraces()
    trace::T
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

function get_demon_parameters(lu::ESARSA, learner, demons, obs, state, action, next_obs, next_state, next_action, env_reward)
    C, next_discounts, _ = get(demons; state_t = obs, action_t = action, state_tp1 = next_obs, action_tp1 = next_action, reward = env_reward)
    target_pis = get_demon_pis(demons, learner.num_actions, state, obs)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)
    C, next_discounts, target_pis, next_target_pis
end


function update!(lu::ESARSA,
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
                 reward) where {M<:AbstractMatrix, LU<:ESARSA}


    weights = learner.model
    λ = lu.lambda
    e = get!(()->zero(weights), lu.e, weights)::typeof(weights)

    # next_target_pis = behaviour_pi_func(next_state, next_obs)
    C, next_discounts, target_pis, next_target_pis = get_demon_parameters(lu, learner, demons, obs, state, action, next_obs, next_state, next_action, reward)
    b_πs = behaviour_pi_func(state, obs)

    discounts = get!(()->zero(next_discounts), lu.prev_discounts, learner)::typeof(next_discounts)


    #NOTE: With exploring starts an action could be taken that goes against b_π with prob 0.
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

    next_preds = learner(next_state)
    pred = learner(state, action)

    Qs = reshape(next_preds, (learner.num_actions, learner.num_demons))'
    td_err = C .+ next_discounts .* sum(next_target_pis .* Qs, dims = 2) - pred
    td_err_across_demons = repeat(vec(td_err), inner=learner.num_actions)

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
    discounts .= next_discounts
end

function update!(lu::ESARSA,
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

    projected_state = learner.feature_projector(obs, action, next_obs)
    projected_state_action = get_active_action_state_vector(projected_state, action, size(learner.feature_projector), learner.num_actions)

    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]


    b_πs = behaviour_pi_func(state, obs)
    #NOTE: With exploring starts an action could be taken that goes against b_π with prob 0.
    ρ_SF = if b_πs[action] == 0
        zeros(length(SF_target_pis[:,action]))
    else
        SF_target_pis[:,action] ./ b_πs[action]
    end
    ρ_reward = if b_πs[action] == 0
        zeros(length(reward_target_pis[:,action]))
    else
        reward_target_pis[:,action] ./ b_πs[action]
    end


    # Update Traces: See update_utils.jl
    update_trace!(lu.trace, e_ψ, active_state_action, λ, SF_discounts, ρ_SF)
    update_trace!(lu.trace, e_w, projected_state_action, λ, reward_discounts, ρ_reward)
    e_nz = e_nz ∪ active_state_action.nzind
    e_w_nz = e_w_nz ∪ projected_state_action.nzind

    pred = ψ * next_active_state_action
    reward_feature_backup = zeros(length(SF_C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += SF_next_target_pis[:,a] .* (ψ * next_possible_active_state_action)
    end

    target = SF_C + SF_next_discounts .* reward_feature_backup
    td_err = target - ψ * active_state_action

    pred_err = reward_C - w * projected_state_action

    # This should always be true as this is immediate next step prediction which is equivalent to having discounts of 0 for all states
    @assert sum(reward_discounts) == 0
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

        # Reward next_discounts is always 0, so don't need projected_next_state_action
        # state_discount_r = -reward_next_discounts * projected_next_state_action'
        state_discount_r = projected_state_action'
        abs_ϕ_w = if λ == 0.0
            abs.(repeat(projected_state_action, outer=(1, length(pred_err)))')
        else
            abs.(e_w)
        end
        z_r = abs_ϕ_w .* max.(abs_ϕ_w, state_discount_r)
        update!(lu.opt, w, e_w, pred_err, z_r, learner.num_tasks, 1)
    elseif lu.opt isa Flux.Descent
        α = lu.opt.eta
        if λ == 0.0
            ψ[:, active_state_action.nzind] .+= (α  * td_err) * active_state_action.nzval'
            w .= w .+ α * pred_err * projected_state_action'
        else
            ψ[:, e_nz] .+= (α  * td_err) .* e_ψ[:, e_nz]
            w[:, e_w_nz] .+= (α  * pred_err) .* e_w[:, e_w_nz]
        end
    else
        update!(lu.opt, ψ, - e_ψ .* td_err)
        update!(lu.opt, w, - e_w .* pred_err)
    end
    discounts .= next_discounts
end


function zero_eligibility_traces!(lu::ESARSA)
    for (k, v) ∈ lu.e
        if eltype(v) <: Integer
            lu.e[k] = Int[]
        else
            v .= 0
        end
    end
end
