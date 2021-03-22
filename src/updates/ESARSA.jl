
mutable struct ESARSA <: LearningUpdate
    lambda::Float64
    e::Array{Float64,2}
    num_gvfs::Int
    num_actions::Int
    alpha::Float64
    trace_type::String
    function ESARSA(lambda, feature_size, num_gvfs, num_actions, alpha, trace_type)
        new(lambda, zeros(num_gvfs * num_actions, feature_size), num_gvfs, num_actions, alpha, trace_type)
    end
end

Base.size(learner::ESARSA) = size(learner.e)

function update!(learner::ESARSA, weights, C, state, action, next_state, next_action, next_target_pis, next_discounts)
    # the eligibility trace is for state-action so need to find the exact state action pair per demon and not just the state
    inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_gvfs]
    if learner.trace_type == "accumulating"
        learner.e[inds,:] .+= state'
    elseif learner.trace_type == "replacing"
        learner.e[inds, state.nzind] .= 1
    else
        throw("Not a valid trace type for ESARSA")
    end

    pred = weights * next_state
    #TODO: FIX assumption that all pseudoterminations occur at the same time.

    target = if next_discounts[1] != 0
        # expected sarsa backup for TB
        Qs = row_order_reshape(pred, (learner.num_gvfs, learner.num_actions))'
        # Target Pi is num_demons  x num_actions
        backup_est_per_demon = if learner.num_gvfs == 1
            sum(Qs * next_target_pis)
        else
            vec(sum((Qs .* next_target_pis ), dims = 2))
        end
        C + next_discounts .* backup_est_per_demon
     else
         C
    end

    # TD error per demon is the td error on Q
    td_err = target - (weights * state)[inds]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    weights .+= learner.alpha * (learner.e .* td_err_across_demons)

    #Broadcast the policy and pseudotermination of each demon across the actions
    # Decay eligibility trace
    learner.e .*= learner.lambda * repeat(next_discounts, inner = learner.num_actions)
end

function get_action_probs(self::ESARSA, state, obs, weights)
    qs = weights * state
    epsilon = 0.2

    # Apply Epsilon Greedy
    m = maximum(qs)
    probs = zeros(length(qs))

    maxes_ind = findall(x-> x==m, qs)
    for ind in maxes_ind
        probs[ind] += (1 - epsilon) / size(maxes_ind)[1]
    end
    probs .+= epsilon/ length(qs)
    return probs
end

function zero_eligibility_traces(self::ESARSA)
    self.e = self.e * 0.0
end
