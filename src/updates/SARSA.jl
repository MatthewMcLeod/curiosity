
# mutable struct ESARSA <: LearningUpdate
#     lambda::Float64
#     e::Array{Float64,2}
#     num_gvfs::Int
#     num_actions::Int
#     alpha::Float64
#     trace_type::String
#     function ESARSA(lambda, feature_size, num_gvfs, num_actions, alpha, trace_type)
#         new(lambda, zeros(num_gvfs * num_actions, feature_size), num_gvfs, num_actions, alpha, trace_type)
#     end
# end

Base.@kwdef mutable struct SARSA{O, T<:AbstractTraceUpdate} <: LearningUpdate
    lambda::Float64
    opt::O
    trace::T = ReplacingTraces()
    e::IdDict = IdDict()
    prev_discounts::IdDict = IdDict()
end

# Base.size(learner::ESARSA) = size(learner.e)

function update!(lu::SARSA,
                 learner::QLearner{M, LU},
                 obs,
                 next_obs,
                 state,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 discount,
                 reward) where {M<:AbstractMatrix, LU<:SARSA}

    if is_terminal
        discount = [0.0]
    end

    weights = learner.model
    λ = lu.lambda
    e = get!(()->zero(weights), lu.e, weights)::typeof(weights)

    inds = get_action_inds(action, learner.num_actions, learner.num_demons)
    state_action_row_ind = inds

    # Only handle on-policy so far
    # NOTE: Trace is being applied in the wrong order for SARSA(lambda)
    ρ = 1
    # update_trace!(lu.trace,
    #               e,
    #               state,
    #               λ,
    #               discount,
    #               ρ,
    #               inds)

    # _accumulate_trace(lu.trace, e, state, inds)

    e[inds, state.nzind] .= 1


    next_pred = learner(next_state, next_action)
    pred = learner(state, action)

    td_err = reward + discount * next_pred' - pred

    Flux.Optimise.update!(lu.opt, weights,  -(e .* td_err))

    e .*= λ * discount .* ρ
end

function zero_eligibility_traces!(lu::SARSA)
    # learner.e .= 0
    for (k, v) ∈ lu.e
        if eltype(v) <: Integer
            lu.e[k] = Int[]
        else
            v .= 0
        end
    end
end
