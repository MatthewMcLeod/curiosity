mutable struct LSTDLearner{AF<:AbstractFloat} <: Learner

    b::Vector{Vector{AF}}
    A_inv::Vector{Matrix{AF}}
    t::Int
    e::Vector{Vector{AF}}

    w::Vector{Matrix{AF}}
    w_real::Vector{Vector{AF}}

    λ::Float64
    γ_t::Vector{Float64}

    feature_size::Int
    num_actions::Int
    num_demons::Int
end

function LSTDLearner{F}(eta, λ, feature_size, num_actions, num_demons) where {F<:Number}
    fsna = feature_size*num_actions
    b = [zeros(F, fsna) for i in 1:num_demons]
    A_inv = [zeros(F, fsna, fsna) .+ F(eta)*I(fsna) for i in 1:num_demons]
    e = [zeros(F, fsna) for i in 1:num_demons]
    w = [zeros(F, num_actions, feature_size) for i in 1:num_demons]
    w_real = [zeros(F, fsna) for i in 1:num_demons]
    LSTDLearner(b,
                A_inv,
                0,
                e,
                w,
                w_real,
                λ,
                zeros(num_demons),
                feature_size,
                num_actions,
                num_demons)
end

LSTDLearner(args...) = LSTDLearner{Float64}(args...)

function get_weights(l::LSTDLearner)
    return [l.w_real[i] for i in 1:l.num_demons]
end

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

function update!(learner::LSTDLearner,
                 demons,
                 obs,
                 next_obs,
                 state,
                 action,
                 next_state,
                 next_action,
                 is_terminal,
                 behaviour_pi_func,
                 reward)


    C, γ_tp1, π_t, π_tp1 =
        get_demon_parameters(learner,
                             demons,
                             obs,
                             state,
                             action,
                             next_obs,
                             next_state,
                             next_action)

    na = learner.num_actions
    fs = learner.feature_size
    λ = learner.λ
    t = learner.t
    μ_t = behaviour_pi_func(state, obs)[action]
    μ_tp1 = behaviour_pi_func(next_state, next_obs)

    ρ_t = π_t ./ μ_t
    ρ_tp1 = π_tp1 ./ μ_tp1'

    γ_t = learner.γ_t

    ϕ_t = state
    ϕ_tp1 = next_state

    x_t = get_active_action_state_vector(
        ϕ_t, action, learner.feature_size, learner.num_actions)
    x_tp1 = get_active_action_state_vector(
        ϕ_tp1, next_action, learner.feature_size, learner.num_actions)

    for gvf ∈ 1:learner.num_demons
        A_inv = learner.A_inv[gvf]
        e = learner.e[gvf]
        b = learner.b[gvf]
        c = C[gvf]

        e .= γ_t[gvf]*λ*ρ_t[gvf] * e + x_t
        b .+= (c*e - b)/(t+1)

        u = sum(ρ_tp1[gvf, a] .* get_active_action_state_vector(ϕ_tp1, a, fs, na) for a ∈ 1:na)
        v = transpose(transpose(x_t - γ_tp1[gvf]*u) * A_inv)

        if t > 0
            scale = (t+1)/t
            vz = dot(v, e)
            Ainv_zvt = (A_inv * e) * transpose(v)
            A_inv .= scale * (A_inv - Ainv_zvt./(t + vz))
        else
            Aev = A_inv*(e*v')
            ve = dot(v, e)
            A_inv .-= Aev/(1 + ve)
        end

        learner.w_real[gvf] .= A_inv * b

    end

    learner.γ_t .= γ_tp1
    learner.t += 1

    # return 0.0, zero(learner.w) .+ t/(t-1)
end

(learner::LSTDLearner)(state, action) = predict(learner::LSTDLearner, state, action)

function predict(learner::LSTDLearner, state, action)
    ϕ = get_active_action_state_vector(state,
                                       action,
                                       learner.feature_size,
                                       learner.num_actions)
    return [dot(learner.w_real[i], ϕ) for i ∈ 1:learner.num_demons]
end

function zero_eligibility_traces!(learner::LSTDLearner)
    for e ∈ learner.e
        e .= 0
    end
end
