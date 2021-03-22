Base.@kwdef mutable struct ESARSA{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    trace::T = ReplacingTraces()
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
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
                 reward,
                 discount,
                 behaviour_pi_func) where {M<:AbstractMatrix, LU<:ESARSA}

    if is_terminal
        discount = [0.0]
    end

    weights = learner.model
    λ = lu.lambda
    e = get!(()->zero(weights), lu.e, weights)::typeof(weights)
    ρ = 1

    next_target_pis = behaviour_pi_func(next_state, next_obs)

    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds

    #NOTE: Cant use elibigility traces as the updates for them do not follow the
    # scaling and then addition (also updating behaviour weights occur between the two steps)
    e[inds, state.nzind] .= 1

    next_preds = learner(next_state)
    pred = learner(state, action)

    td_err = reward + discount * sum(next_target_pis .* next_preds) - pred
    Flux.Optimise.update!(lu.opt, weights,  -(e .* td_err))

    e .*= λ * discount .* ρ
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
