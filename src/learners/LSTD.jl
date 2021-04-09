mutable struct LSTDLearner{AF<:AbstractFloat} <: Learner

    update::LearningUpdate
    b::Vector{Vector{AF}}
    A_inv::Vector{Matrix{AF}}
    t::Int
    e::Vector{Vector{AF}}

    w::Vector{Matrix{AF}}
    w_real::Vector{Vector{AF}}

    feature_size::Int
    num_actions::Int
    num_demons::Int
end

function LSTDLearner{F}(update, eta, feature_size, num_actions, num_demons) where {F<:Number}
    fsna = feature_size*num_actions
    b = [zeros(F, fsna) for i in 1:num_demons]
    A_inv = [zeros(F, fsna, fsna) .+ F(eta)*I(fsna) for i in 1:num_demons]
    e = [zeros(F, fsna) for i in 1:num_demons]
    w = [zeros(F, num_actions, feature_size) for i in 1:num_demons]
    w_real = [zeros(F, fsna) for i in 1:num_demons]
    LSTDLearner(update,
                b,
                A_inv,
                0,
                e,
                w,
                w_real,
                feature_size,
                num_actions,
                num_demons)
end

LSTDLearner(args...) = LSTDLearner{Float64}(args...)
update(l::LSTDLearner) = l.update

function get_demon_parameters(learner::LSTDLearner,
                              demons,
                              obs,
                              state,
                              action,
                              next_obs,
                              next_state,
                              next_action)
    C, next_discounts, π_t = get(demons, obs, action, next_obs, next_action)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)
    C, next_discounts, π_t, next_target_pis
end

function get_weights(l::LSTDLearner)
    return l.w_real
end

(learner::LSTDLearner)(state, action) = predict(learner::LSTDLearner, state, action)

function predict(learner::LSTDLearner, state, action)
    ϕ = get_active_action_state_vector(state,
                                       action,
                                       learner.feature_size,
                                       learner.num_actions)
    return [dot(learner.w_real[i], ϕ) for i ∈ 1:learner.num_demons]
end
