using GVFHordes
import ProgressMeter


StatsBase.sample(rng::Random.AbstractRNG, p::GVFHordes.GVFParamFuncs.FunctionalPolicy, s, actions) =
    sample(rng, Weights([p.func(state_t = s, action_t = a) for a in actions]))


"""
    monte_carlo_return

Perform a monte_carlo_return using the provided environment, gvf, and start_state. For an environment to be compatible it must
implement `reset!(env, state)`

The interface is limited in that it can't do rollouts on compositional GVFs.
"""
function monte_carlo_return(env,
                            gvf,
                            start_state,
                            start_action,
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
        # next_action = StatsBase.sample(rng, GVFHordes.policy(gvf), cur_state, get_actions(env))
        next_action = start_action

        while cumulative_gamma > γ_thresh &&
            step < max_steps &&
            term == false

            # Take action
            action = next_action
            next_state, r, term = MinimalRLCore.step!(env, action, rng)

            # Get next action for GVFs
            next_action = StatsBase.sample(rng, GVFHordes.policy(gvf), next_state, get_actions(env))

            # Update Return
            # c, γ, pi_prob = get(gvf, cur_state, action, next_state, next_action, nothing)
            c, γ, pi_prob = get(gvf; state_t = cur_state, action_t = action, state_tp1 = next_state, action_tp1 = next_action)
            returns[ret] += cumulative_gamma*c
            cumulative_gamma *= γ*(1-term)

            cur_state = next_state
            step += 1
        end
    end

    return returns
end

function monte_carlo_returns(env, gvf, start_states, actions, num_returns, γ_thresh, max_steps=Int(1e6), rng=Random.GLOBAL_RNG)

    states_actions = zip(start_states, actions)
    ret = Vector{Vector{Float64}}(undef, length(states_actions))
    prg_meter = ProgressMeter.Progress(length(states_actions))
    Threads.@threads for i ∈ 1:length(states_actions)
        t_env = deepcopy(env)
        ret[i] = monte_carlo_return(t_env, gvf, start_states[i], actions[i], num_returns, γ_thresh, max_steps)
        ProgressMeter.next!(prg_meter)
    end
    ret

end

function monte_carlo_returns(env, gvf, start_states, actions, num_returns, γ_thresh, max_steps=Int(1e6), rng=Random.GLOBAL_RNG; agg)

    # states_actions = zip(start_states, actions)
    ret = Vector{Float64}(undef, length(start_states))
    prg_meter = ProgressMeter.Progress(length(start_states))
    Threads.@threads for i ∈ 1:length(start_states)
        t_env = deepcopy(env)
        ret[i] = agg(monte_carlo_return(t_env, gvf, start_states[i], actions[i], num_returns, γ_thresh, max_steps))
        ProgressMeter.next!(prg_meter)
    end
    ret

end
