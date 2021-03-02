using GVFHordes


StatsBase.sample(rng::Random.AbstractRNG, p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(rng, Weights([p.func(s, a) for a in actions]))


"""
    monte_carlo_return

Perform a monte_carlo_return using the provided environment, gvf, and start_state. For an environment to be compatible it must
implement `reset!(env, state)`

The interface is limited in that it can't do rollouts on compositional GVFs.
"""
function monte_carlo_return(env,
                            gvf,
                            start_state,
                            num_returns,
                            γ_thresh=1e-6,
                            max_steps=Int(1e6),
                            rng=Random.GLOBAL_RNG)

    returns = zeros(num_returns)

    for ret in 1:num_returns
        step = 0
        term = false
        cumulative_gamma = 1.0

        cur_state = start!(env, start_state)
        next_action = StatsBase.sample(rng, GVFHordes.policy(gvf), cur_state, get_actions(env))

        while cumulative_gamma > γ_thresh &&
            step < max_steps &&
            term == false

            # Take action
            action = next_action
            next_state, r, term = MinimalRLCore.step!(env, action, rng)


            # Get next action for GVFs
            next_action = StatsBase.sample(rng, GVFHordes.policy(gvf), next_state, get_actions(env))

            # Update Return
            c, γ, pi_prob = get(gvf, cur_state, action, next_state, next_action, nothing)
            returns[ret] += cumulative_gamma*c
            cumulative_gamma *= γ*(1-term)

            cur_state = next_state
            step += 1
        end
    end

    return returns
end

monte_carlo_returns(env, gvf, start_states, num_returns, γ_thresh, max_steps=Int(1e6), rng=Random.GLOBAL_RNG) =
    [monte_carlo_return(env, gvf, st, num_returns, γ_thresh, max_steps) for st in start_states]
