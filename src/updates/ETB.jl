
# Implementation taken from p316 RL Textbook and incorporating TB
Base.@kwdef mutable struct ETB{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    hdpi::HordeDPI
    clip_threshold::Float64
    normalize_interest::Bool
    trace::T = AccumulatingTraces()
    e::IdDict = IdDict()
    followon::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
    emphasis_logging::Any = zeros(4) #I'm lazy and this is used for tracking hard to recalculate values
    rho_logging::Any = zeros(4) #I'm lazy and this is used for tracking hard to recalculate values
end

function get_demon_parameters(lu::ETB, learner, demons, obs, state, action, next_obs, next_state, next_action, env_reward)
    # C, next_discounts, _ = get(demons, obs, action, next_obs, next_action)
    C, next_discounts, _ = get(demons; state_t = obs, action_t = action, state_tp1 = next_obs, action_tp1 = next_action, reward = env_reward)
    target_pis = get_demon_pis(demons, learner.num_actions, state, obs)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)
    C, next_discounts, target_pis, next_target_pis
end


function update!(lu::ETB,
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
                 env_reward) where {M<:AbstractMatrix, LU<:ETB}

    weights = learner.model
    λ = lu.lambda

    behaviour_pis = behaviour_pi_func(state, obs)
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
    followon = get!(()->zeros(learner.num_actions), lu.followon, weights)::typeof(zeros(learner.num_actions))


    #Broadcast the policy and pseudotermination of each demon across the actions
    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds

    # interest = get_interest(learner, lu, obs, action)

    # using dpi for interest instead
    interest = lu.hdpi(obs, action)
    if (lu.normalize_interest)
        interest = [ceil(i) for i in interest]
    end

    # getting IS ratio
    if (behaviour_pis[action] == 0)
        ρ = zeros(size(target_pis[:, action]))
    else
        ρ = target_pis[:, action] / behaviour_pis[action]
    end

    # accumulating followon
    followon[:] .+= interest

    # Get Emphasis vector - size is # demons
    # Correcting with ρ in the beginning since it is the action-value emphasis rather than the state-value emphasis
    emphasis = ρ .* (λ * behaviour_pis[action] * interest + (1 - λ * behaviour_pis[action]) * followon)

    # I'm lazy, here we're just logging the emphasis and rho as variables.
    lu.emphasis_logging = emphasis
    lu.rho_logging = ρ

    clamp!(emphasis, 0, lu.clip_threshold)

    # Update eligibility trace
    update_trace!(lu.trace,
                  e,
                  state,
                  λ,
                  repeat(discounts, inner = learner.num_actions),
                  repeat(target_pis[:,action], inner = learner.num_actions),
                  inds;
                  emphasis=emphasis)

    # decaying followon
    followon[:] .*= ρ .* next_discounts

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

function update!(lu::ETB,
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

    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]

    # Followon
    followon = get!(()->zeros(learner.num_demons), lu.followon, weights)::typeof(zeros(learner.num_demons))

    # Setting interest to constant ones for now for each demon
    # interest = get_interest(learner, lu, obs, action)

    demon_interest = lu.hdpi(obs, action)

    if (lu.normalize_interest)
        demon_interest = [ceil(i) for i in demon_interest]
    end

    num_repeat = convert(Integer, (learner.num_demons - learner.num_tasks) / learner.num_tasks)
    if (rem(learner.num_demons - learner.num_tasks, learner.num_tasks) != 0)
        println("The number to repeat for SF demons aren't even with the number of tasks for SR. Something might be wrong :(")
    end

    SF_interest = repeat(demon_interest, inner=num_repeat)
    reward_interest = demon_interest

    interest = vcat(reward_interest, SF_interest)

    # getting IS ratio
    behaviour_pis = behaviour_pi_func(state, obs)
    if (behaviour_pis[action] == 0)
        ρ = zeros(size(target_pis[:, action]))
    else
        ρ = target_pis[:, action] / behaviour_pis[action]
    end
    reward_ρ, SF_ρ = ρ[1:learner.num_tasks], ρ[learner.num_tasks+1:end]

    # accumulating followon
    followon[:] .+= interest

    # Get Emphasis vector - size is # demons
    # Correcting with ρ in the beginning since it is the action-value emphasis rather than the state-value emphasis
    emphasis = ρ .* (λ * behaviour_pis[action] * interest + (1 - λ * behaviour_pis[action]) * followon)

    # I'm lazy, here we're just logging the emphasis and rho as variables.
    lu.emphasis_logging = emphasis
    lu.rho_logging = ρ


    clamp!(emphasis, 0, lu.clip_threshold)
    reward_emphasis, SF_emphasis = emphasis[1:learner.num_tasks], emphasis[learner.num_tasks+1:end]


    # Update Traces: See update_utils.jl
    update_trace!(lu.trace, e_ψ, active_state_action, λ, SF_discounts, SF_target_pis[:, action]; emphasis=SF_emphasis)
    update_trace!(lu.trace, e_w, projected_state, λ, reward_discounts, reward_target_pis[:, action]; emphasis=reward_emphasis)
    # e_nz = e_nz ∪ active_state_action.nzind
    # e_w_nz = e_w_nz ∪ projected_state_action.nzind
    if λ == 0.0
        e_nz = active_state_action.nzind
        e_w_nz = projected_state.nzind
    else
        e_nz = e_nz ∪ active_state_action.nzind
        e_w_nz = e_w_nz ∪ projected_state.nzind
    end

    # decaying followon
    followon[:] .*= ρ .* next_discounts


    pred = ψ * next_active_state_action
    reward_feature_backup = zeros(length(SF_C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += SF_next_target_pis[:,a] .* (ψ * next_possible_active_state_action)
    end

    target = SF_C + SF_next_discounts .* reward_feature_backup
    td_err = target - ψ * active_state_action

    pred_err = reward_C - w * projected_state

    # This should always be true as this is immediate next step prediction which is equivalent to having discounts of 0 for all states
    @assert sum(reward_discounts) == 0
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

        # reward state discount is always  0 so no need for next_state_action.
        # state_discount_r = -reward_next_discounts * projected_next_state_action'
        state_discount_r = projected_state'
        abs_ϕ_w = if λ == 0.0
            abs.(repeat(projected_state, outer=(1, length(pred_err)))')
        else
            abs.(e_w)
        end
        z_r = abs_ϕ_w .* max.(abs_ϕ_w, state_discount_r)
        update!(lu.opt, w, e_w, pred_err, z_r, learner.num_tasks, 1)
    elseif lu.opt isa Flux.Descent
        α = lu.opt.eta
        if λ == 0.0
            ψ[:, active_state_action.nzind] .+= (α  * td_err) * active_state_action.nzval'
            w .= w .+ α * pred_err * projected_state'
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


function zero_eligibility_traces!(lu::ETB)
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
