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

function row_order_reshape(A, reshape_dims)
    #I dont think Julia natively supports row based reshape
    # https://github.com/JuliaLang/julia/issues/20311
    return transpose(reshape(transpose(A), reshape_dims...))
end

function update!(learner::TB, weights, C, discounts, state, action, target_pis, next_state, next_action, next_target_pis)
    # Update eligibility trace
    # target_pis = [get(learner.gvfs[gvf_i].policy, state, action) for gvf_i in 1:learner.num_demons]
    # target_pis = zeros(length(agent.demons), agent.num_actions)
    # for (i,a) in enumerate(1:agent.num_actions)
    #     _, _, pi = get(agent.demons, obs, a, next_obs, next_action)
    #     target_pis[:,i] = pi
    # end
    # next_target_pis = zeros(length(agent.demons), agent.num_actions)
    # for (i,a) in enumerate(1:agent.num_actions)
    #     _, _, pi = get(agent.demons, obs, a, next_obs, next_action)
    #     next_target_pis[:,i] = pi
    # end

    #Broadcast the policy and pseudotermination of each demon across the actions
    learner.e .*= learner.lambda * repeat(discounts, inner = learner.num_actions) .* repeat(target_pis[:,action], inner = learner.num_actions)

    # the eligibility trace is for state-action so need to find the exact state action pair per demon
    inds = [action + (i-1)*learner.num_actions for i in 1:learner.num_actions]
    learner.e[inds,:] .+= state'

    pred = weights * next_state
    #TODO: FIX assumption that all pseudoterminations occur at the same time.
    target = if discounts[1] != 0
        # expected sarsa backup for TB
        Qs = row_order_reshape(pred, (learner.num_demons, learner.num_actions))
        # Target Pi is num_demons  x num_actions
        backup_est_per_demon = vec(sum((Qs .* next_target_pis ), dims = 2))
        C + discounts .* backup_est_per_demon
     else
         C
     end
     # TD error per demon is the td error on Q
    td_err = target - (weights * state)[inds]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    # How to efficiently apply gradients back into weights? Should we move linear regression to Flux/Autograd?
    # TODO: Seperate optimizer and learning algo

    # if C[2] == 1.0
    #     println("End with discount: ", discounts[2])
    #     println("state: ", state, " action: ", action)
    #     println("Esimate: ", weights[7,3:5])
    #     asdf=1
    # end
    if td_err_across_demons[7] < -0.1
        println("How is td err on constant negative?", td_err_across_demons[7])
        println("Target policy assigns prob: ", target_pis[2,:])
        println("Next Target policy assigns prob: ", next_target_pis[2,:])
        println("state: ", state, " action: ", action, " next_state ", next_state, " next_action: ", next_action)
        println("Esimate: ", weights[7,3:5])
        println("Target ", target)
        println()
    end
    weights .= weights + learner.alpha * (learner.e .* td_err_across_demons)

end

function zero_eligibility_traces!(learner::TB)
    learner.e .= 0
end
