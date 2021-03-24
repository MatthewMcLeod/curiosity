Base.@kwdef mutable struct SARSA{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    trace::T = ReplacingTraces()
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

function update!(lu::SARSA,
                 learner::QLearner{M, LU},
                 demons,
                 obs,
                 next_obs,
                 state,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 discount,
                 behaviour_pi_func,
                 reward) where {M<:AbstractMatrix, LU<:SARSA}

    if is_terminal
        discount = [0.0]
    end

    weights = learner.model
    λ = lu.lambda
    e = get!(()->zero(weights), lu.e, weights)::typeof(weights)

    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds


    #NOTE: Cant use elibigility traces as the updates for them do not follow the
    # scaling and then addition (also updating behaviour weights occur between the two steps)
    ρ = 1
    e[inds, state.nzind] .= 1


    next_pred = learner(next_state, next_action)
    pred = learner(state, action)

    td_err = reward + discount * next_pred' - pred

    Flux.Optimise.update!(lu.opt, weights,  -(e .* td_err))

    e .*= λ * discount .* ρ
end

function zero_eligibility_traces!(lu::SARSA)
    for (k, v) ∈ lu.e
        if eltype(v) <: Integer
            lu.e[k] = Int[]
        else
            v .= 0
        end
    end
end
