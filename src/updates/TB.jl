

Base.@kwdef mutable struct TB{O} <: LearningUpdate
    lambda::Float64
    opt::O
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

get_action_inds(action, num_actions, num_gvfs) = [action + (i-1)*num_actions for i in 1:num_gvfs]

function get_demon_parameters(lu::TB, learner, demons, obs, state, action, next_obs, next_state, next_action)
    C, next_discounts, _ = get(demons, obs, action, next_obs, next_action)
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
                 behaviour_pi_func) where {M<:AbstractMatrix, LU<:TB}

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
                             next_action)

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
end

function update!(lu::TB,
                 learner::SRLearner,
                 demons,
                 obs,
                 next_obs,
                 state::SparseVector,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 behaviour_pi_func)

    
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
                             next_action)

    discounts = get!(lu.prev_discounts, learner, zero(next_discounts))::typeof(next_discounts)


    next_active_state_action = get_active_action_state_vector(next_state, next_action,length(next_state), learner.num_actions)
    active_state_action = get_active_action_state_vector(state, action, length(state), learner.num_actions)
    
    (reward_C, SF_C) = C[1:learner.num_tasks] , C[learner.num_tasks + 1:end]
    (reward_discounts, SF_discounts) = discounts[1:learner.num_tasks], discounts[learner.num_tasks+1:end]
    (reward_next_discounts, SF_next_discounts) = next_discounts[1:learner.num_tasks], next_discounts[learner.num_tasks+1:end]
    (reward_target_pis, SF_target_pis) = target_pis[1:learner.num_tasks,:], target_pis[learner.num_tasks+1:end,:]
    (reward_next_target_pis, SF_next_target_pis) = next_target_pis[1:learner.num_tasks,:], next_target_pis[learner.num_tasks+1:end, :]

    
    pred = ψ * next_active_state_action
    reward_feature_backup = zeros(length(SF_C))
    for a in 1:learner.num_actions
        next_possible_active_state_action = get_active_action_state_vector(next_state, a, length(next_state), learner.num_actions)
        reward_feature_backup += SF_next_target_pis[:,a] .* (ψ * next_possible_active_state_action)
    end

    target = SF_C + SF_next_discounts .* reward_feature_backup
    td_err = target - ψ * active_state_action

    pred_err = reward_C - w * active_state_action
    #td_err is (336x1)
    # TD err is applied across rows

    if lu.opt isa Auto
        throw("SR + TB + Auto not implemented")
    elseif lu.opt isa Flux.Descent
        α = lu.opt.eta
        ψ[:, active_state_action.nzind] .+= (α  * td_err) * active_state_action.nzval'
        w .= w .+ α * pred_err * active_state_action'
    else
        update!(lu.opt, ψ, -td_err * active_state_action')
        update!(lu.opt, w, -pred_err * active_state_action')
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
