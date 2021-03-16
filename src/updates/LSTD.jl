mutable struct LSTD <: Learner
    feature_size::Int
    num_actions::Int
    
    b::Vector{Float64}
    A_inv::Matrix{Float64}
    t::Int

    w::Matrix{Float64}
    w_real::Vector{Float64}
    
    λ::Float64
    γ_t::Float64
    γ::GVFPseudoTermination
    π::GVFPolicy
    
    e::Vector{Float64}
    gvf_i::Int

    replay_type::String
    optimizer::Optimizer
    er::AbstractReplay

    LSTD() = new()
end

function init!(self::LSTD;
               eta=0.001,
               feature_size=1,
               num_actions=1,
               lambda=0.0,
               gamma,
               policy,
               gvf_i)
    
    self.feature_size = feature_size
    self.num_actions = num_actions

    self.b = zeros(feature_size*num_actions)
    self.A_inv = zeros(feature_size*num_actions, feature_size*num_actions)
    self.A_inv .+= eta*I(feature_size*num_actions)

    self.t = 0

    self.w = zeros(num_actions, feature_size)
    self.w_real = zeros(num_actions*feature_size)

    self.λ = lambda
    self.γ_t = 1.0
    self.γ = gamma
    self.π = policy
    
    self.e = zero(self.w_real)

    self.gvf_i = gvf_i

    self.replay_type = "None"
end

function update!(lstd::LSTD,
                 reward,
                 action,
                 next_action,
                 observation,
                 next_observation,
                 e,
                 is_replay,
                 optimizer_params,
                 agent)

    t = lstd.t

    γ_tp1 = gamma(lstd.γ, next_observation)
    γ_t = gamma(lstd.γ, observation)
    ρ_tp1 = policy(lstd.π, next_observation)
    ρ_t = policy(lstd.π, observation)

    c = reward

    ϕ_t = state_construction(agent, observation)
    ϕ_tp1 = state_construction(agent, next_observation)
    
    x_t = get_active_action_state_vector(
        ϕ_t, action, lstd.feature_size, lstd.num_actions)
    x_tp1 = get_active_action_state_vector(
        ϕ_tp1, next_action, lstd.feature_size, lstd.num_actions)

    lstd.e .= γ_t*lstd.λ*ρ_t[action] * lstd.e + x_t
    lstd.b .+= (c*lstd.e - lstd.b)/(t+1)
    
    u = sum(ρ_tp1[a].*get_active_action_state_vector(ϕ_tp1, a, lstd.feature_size, lstd.num_actions) for a ∈ 1:lstd.num_actions)
    v = transpose(transpose(x_t - γ_tp1*u) * lstd.A_inv)

    if t > 0
        scale = (t+1)/t
        vz = dot(v, lstd.e)
        Ainv_zvt = (lstd.A_inv * lstd.e) * transpose(v)
        lstd.A_inv .= scale * (lstd.A_inv - Ainv_zvt./(t + vz))
    else
        Aev = lstd.A_inv*(lstd.e*v')
        ve = dot(v, lstd.e)
        lstd.A_inv .-= Aev/(1 + ve)
    end
    
    lstd.w_real .= lstd.A_inv * lstd.b
    
    lstd.γ_t = γ_t
    lstd.t += 1
    # println(lstd.t)
    return 0.0, zero(lstd.w) .+ t/(t-1)
end

function predict(lstd::LSTD, agent, observation, action)
    ϕ = get_active_action_state_vector(
        state_construction(agent, observation),
        action,
        lstd.feature_size,
        lstd.num_actions)
    return dot(lstd.w_real, ϕ)
end

function zero_eligibility_traces(self::LSTD)
    self.e = self.e * 0.0
end

update_eligibility_traces!(
    self::LSTD,
    args...) = nothing
