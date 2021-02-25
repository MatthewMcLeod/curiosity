mutable struct TBAuto <: Learner
    lambda::Float64
    mu::Float64
    e::Array{Float64,2}
    alpha::Array{Float64,2}
    h::Array{Float64,2}
    n::Array{Float64,2}
    z::Array{Float64,2}
    num_demons::Int
    num_actions::Int
    tau::Int
    max_update::Int

    function TBAuto(lambda, feature_size, num_demons, num_actions, alpha, alpha_init)
        new(lambda,
        alpha,
        zeros(num_actions * num_demons, feature_size),
        ones(num_actions * num_demons, feature_size) * alpha_init,
        zeros(num_actions * num_demons, feature_size),
        ones(num_actions * num_demons, feature_size),
        ones(num_actions * num_demons, feature_size),
        num_demons,
        num_actions,
        10000,
        1)
    end
end

function update!(learner::TBAuto,
                 weights,
                 C,
                 state,
                 action,
                 target_pis,
                 discounts,
                 next_state,
                 next_action,
                 next_target_pis,
                 next_discounts)
    
    # Update eligibility trace
    #Broadcast the policy and pseudotermination of each demon across the actions
    learner.e .*= learner.lambda * repeat(discounts, inner = learner.num_actions) .* repeat(target_pis[:,action], inner = learner.num_actions)

    # the eligibility trace is for state-action so need to find the exact state action pair per demon and not just the state
    state_action_row_ind = [action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
    learner.e[state_action_row_ind,:] .+= state'

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
    td_err = target - (weights * state)[state_action_row_ind]
    td_err_across_demons = repeat(td_err, inner=learner.num_actions)

    # How to efficiently apply gradients back into weights? Should we move linear regression to Flux/Autograd?
    # Changing for Auto:
    phi = learner.e
    abs_phi = abs.(phi)

    #TODO: Discounts is for prev_observation, not next observation!! FIX! (Doesnt matter for us since episode terminates, but will cause bug in future)
    # How should should the z be calculated with state-action rather state?
    state_discount = zeros(size(phi)...)
    state_discount[state_action_row_ind,:] .+= state'
    next_state_action_row_ind = [next_action + (i-1)*learner.num_actions for i in 1:learner.num_demons]
    state_discount[next_state_action_row_ind,:] .-= next_discounts * next_state'

    z = abs_phi .* max.(state_discount, abs_phi)

    # Now the Auto Algorithm
    learner.n .= learner.n + (1 / learner.tau) * (learner.alpha .* abs_phi) .* (abs.(td_err_across_demons .* learner.h .* phi) .- learner.n)
    # for all phi != 0 do some stuff
    active_phi_ind = findall(!iszero, phi)
    # Since there are multiple demons in this weight matrix, we want to weight the alpha change by the appropriate td error.
    # I think the easiest is to make the td_error the same size as phi and then non-zero phi indices can just access the appropriate td_err.
    td_err_across_demons_and_states = repeat(td_err_across_demons, 1, size(phi)[2])
    # Clamping instead of min (according to write up. Clamping is a future revision that will be put into Auto)
    alpha_change = clamp.(td_err_across_demons_and_states[active_phi_ind] .* phi[active_phi_ind] .* learner.h[active_phi_ind] ./ learner.n[active_phi_ind], -1, 1)
    learner.alpha[active_phi_ind] = learner.alpha[active_phi_ind] .* exp.(learner.mu * alpha_change)
    learner.alpha[active_phi_ind] = clamp.(learner.alpha[active_phi_ind], 1e-6, 1 ./ abs_phi[active_phi_ind])


    # Check the norm to make sure it isn't too high
    if sum(learner.alpha .* z) > 1
        non_zero_z_ind = findall(!iszero, z)
        temp = min.(learner.alpha, ones(size(z)) ./ sum(abs.(z)))
        learner.alpha[non_zero_z_ind] = temp[non_zero_z_ind]
    end

    weights .= weights + td_err_across_demons .* learner.alpha .* phi
    learner.h .= learner.h .* (1 .- learner.alpha .* abs_phi) .+ td_err_across_demons .* learner.alpha .* phi
end

function zero_eligibility_traces!(learner::TBAuto)
    learner.e .= 0
end

