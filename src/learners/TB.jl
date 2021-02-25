mutable struct TB <: Learner
    lambda::Float64
    e::Array{Float64,2}
    num_demons::Int
    num_actions::Int
    alpha::Float64
    function TB(lambda, feature_size, num_demons, num_actions, alpha)
        new(lambda, zeros(num_actions * num_demons, feature_size), num_demons, num_actions, alpha)
    end
end



function update!(learner::TB, weights, C, state, action, target_pis, discounts, next_state, next_action, next_target_pis, next_discounts)
    # Update eligibility trace
    #Broadcast the policy and pseudotermination of each demon across the actions
    learner.e .*= learner.lambda * repeat(discounts, inner = learner.num_actions) .* repeat(target_pis[:,action], inner = learner.num_actions)

    # the eligibility trace is for state-action so need to find the exact state action pair per demon and not just the state
    inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
    learner.e[inds,:] .+= state'

    pred = weights * next_state
    #TODO: FIX assumption that all pseudoterminations occur at the same time.
    target = if next_discounts[1] != 0
        # expected sarsa backup for TB
        Qs = row_order_reshape(pred, (learner.num_demons, learner.num_actions))
        # Target Pi is num_demons  x num_actions
        backup_est_per_demon = vec(sum((Qs .* next_target_pis ), dims = 2))
        C + next_discounts .* backup_est_per_demon
     else
         C
     end
     # TD error per demon is the td error on Q
    td_err = target - (weights * state)[inds]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    # How to efficiently apply gradients back into weights? Should we move linear regression to Flux/Autograd?
    # TODO: Seperate optimizer and learning algo
    weights .= weights + learner.alpha * (learner.e .* td_err_across_demons)
end

function zero_eligibility_traces!(learner::TB)
    learner.e .= 0
end

function get_weights(learner::TB, weights)
    return weights
end
