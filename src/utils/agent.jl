
using Flux

function get_optimizer(parsed::Dict, prefix="")
    opt_key = prefix == "" ? "opt" : join([prefix, "opt"], "_")
    opt_string = parsed[opt_key]
    get_optimizer(opt_string, parsed, prefix)
end

function get_optimizer(opt_string, parsed::Dict, prefix)
    opt_type = if opt_string == "Auto"
        Auto
    else
        getproperty(Flux, Symbol(opt_string))
    end
    _init_optimizer(opt_type, parsed, prefix)
end

_init_optimizer(opt, ::Dict) =
    throw("$(string(opt)) optimizer initialization not found.")

function _init_optimizer(opt_type::Union{Type{Descent}, Type{ADAGrad}, Type{ADADelta}}, parsed::Dict, prefix = "")
    eta_str = prefix == "" ? "eta" : join([prefix, "eta"], "_")
    try
        η = parsed[eta_str]
        opt_type(η)
    catch
        throw("$(opt_type) needs: $(eta_str) (float).")
    end
end

function _init_optimizer(opt_type::Union{Type{RMSProp},
                                         Type{Momentum},
                                         Type{Nesterov}},
                         parsed::Dict, prefix="")

    eta_str = prefix == "" ? "eta" : join([prefix, "eta"], "_")
    rho_str = prefix == "" ? "rho" : join([prefix, "rho"], "_")
    try
        η = parsed[eta_str]
        ρ = parsed[rho_str]
        opt_type(η, ρ)
    catch
        throw("$(opt_type) needs: $(eta_str) (float), and $(rho_str) (float).")
    end
end

function _init_optimizer(opt_type::Union{Type{ADAM},
                                         Type{RADAM},
                                         Type{NADAM},
                                         Type{AdaMax},
                                         Type{AMSGrad}},
                         parsed::Dict, prefix="")

    eta_str = prefix == "" ? "eta" : join([prefix, "eta"], "_")
    beta_str = prefix == "" ? "beta" : join([prefix, "beta"], "_")
    try
        η = parsed[eta_str]
        β = if beta_str ∈ keys(parsed)
            parsed[beta_str]
        else
            (parsed[join([beta_str, "m"], "_")],
             parsed[join([beta_str, "v"], "_")])
        end
        opt_type(η, β)
    catch
        throw("$(opt_type) needs: $(eta_str) (float), and $(beta_str) ((float, float)), or (beta_m, beta_v)).")
    end
end

function _init_optimizer(opt_type::Union{Type{Auto}}, parsed::Dict, prefix="")
    α_str = prefix == "" ? "alpha" : join([prefix, "eta"], "_")
    α_init_str = prefix == "" ?  "alpha_init" : join([prefix, "alpha_init"], "_")
    try
        α = parsed[α_str]
        α_init = parsed[α_init_str]
        opt_type(α, α_init)
    catch
        throw("$(opt_type) needs: $(α_str) (float), and $(α_init_str) (float).")
    end
end

function get_linear_learner(parsed::Dict,
                            feature_size::Int,
                            num_actions::Int,
                            num_demons::Int,
                            num_tasks::Int,
                            prefix="",
                            feature_projector = nothing)

    learner_key = prefix == "" ? "learner" : join([prefix, "learner"], "_")
    opt = get_optimizer(parsed, prefix)
    lu = get_learning_update(parsed, opt, prefix)
    w_init_key = prefix == "" ? "w_init" : join([prefix, "w_init"], "_")
    w_init = w_init_key in keys(parsed) ? parsed[w_init_key] : 0

    learner_str = parsed[learner_key]
    demon_learner = if learner_str ∈ ["Q", "QLearner", "q"]
        LinearQLearner(lu,
                       feature_size,
                       num_actions,
                       num_demons,
                       w_init)
    elseif learner_str ∈ ["SR", "SRLearner", "sr"]
        SRLearner(lu,
                  feature_size,
                  num_demons,
                  num_actions,
                  num_tasks,
                  feature_projector,
                  w_init)
    elseif learner_str ∈ ["GPI", "gpi"]
        GPI(lu,
            feature_size,
            num_demons,
            num_actions,
            num_tasks,
            feature_projector,
            w_init)
    elseif learner_str ∈ ["LSTD", "LSTDLearner"]
        LSTDLearner(lu,
                    parsed[eta_str],
                    feature_size,
                    num_actions,
                    num_demons)
    elseif learner_str ∈ ["NoLearner", "nolearner"]
        NoLearner(collect(1:num_actions), num_demons)
    else
        throw(ArgumentError("Not a valid demon learner"))
    end
end

function get_linear_learner(parsed::Dict,
                            feature_size,
                            num_actions,
                            demons,
                            prefix="",
                            feature_projector=nothing)

    # Don't want to change call now. num_tasks only exists for SRHorde demons.
    num_tasks = if !hasproperty(demons, :num_tasks)
        0
    else
        demons.num_tasks
    end
    return get_linear_learner(parsed::Dict,
        feature_size,
        num_actions,
        length(demons),
        num_tasks,
        prefix,
        feature_projector)
end

function get_learning_update(parsed::Dict, opt, prefix="")
    lu_key = prefix == "" ? "update" : join([prefix, "update"], "_")
    lu = getproperty(Curiosity, Symbol(parsed[lu_key]))
    _init_learning_update(lu, opt, parsed, prefix)
end

_init_learning_update(lu_type, args...) =
    throw("$(string(lu_type)) does not have an init function.")

function _init_learning_update(lu_type::Union{Type{TabularRoundRobin}}, args...)
    lu_type()
end

function _init_learning_update(lu_type::Union{Type{TB}},
                               opt,
                               parsed::Dict,
                               prefix)
    λ_str = prefix == "" ? "lambda" : join([prefix, "lambda"], "_")
    try
        λ = parsed[λ_str]
        lu_type(lambda=λ, opt=opt)
    catch
        throw("$(lu_type) needs: $(λ_str) (float).")
    end
end

function _init_learning_update(lu_type::Union{Type{ESARSA}, Type{SARSA}},
                                opt,
                                parsed::Dict,
                                prefix)
    λ_str = prefix == "" ? "lambda" : join([prefix, "lambda"], "_")
    trace_str = prefix == "" ? "trace" : join([prefix, "trace"], "_")
    trace = try
        getproperty(Curiosity, Symbol(parsed[trace_str]))
    catch
        throw("$(lu_type) needs trace type")
    end

    try
        λ = parsed[λ_str]
        lu_type(lambda=λ, opt=opt, trace=trace())
    catch
        throw("$(lu_type) needs: $(λ_str) (float).")
    end
end

function get_exploration_strategy(parsed, action_set)
    if parsed["exploration_strategy"] == "epsilon_greedy"
        EpsilonGreedy(parsed["exploration_param"])
    elseif parsed["exploration_strategy"] == "epsilon_greedy_decay"
        ϵGreedyDecay(
            parsed["ϵ_range"],
            parsed["decay_period"],
            parsed["warmup_steps"],
            action_set
        )
    else
        throw(ArgumentError("Not a Valid Exploration Strategy"))
    end
end

Base.@kwdef struct NoLearner <: Learner
    action_set::Array{Int,1}
    num_demons::Int
end

Curiosity.update!(learner::NoLearner, args...) = nothing

Base.get(π::NoLearner; state_t, action_t, kwargs...) =
    get_action_probs(π, state_t, nothing)[action_t]

function Curiosity.get_action_probs(l::NoLearner, features, state)
    return ones(l.num_demons,length(action_set)) ./ length(action_set)
end
(l::NoLearner)(ϕ,a) = zeros(l.num_demons)
(l::NoLearner)(ϕ) = [zeros(l.num_demons) for a in l.action_set]
function zero_eligibility_traces!(l::NoLearner) nothing end
