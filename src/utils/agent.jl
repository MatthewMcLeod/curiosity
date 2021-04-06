
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
        η = parsed[beta_str]
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
#
function get_linear_learner(parsed::Dict,
                            feature_size::Int,
                            num_actions::Int,
                            num_demons::Int,
                            num_tasks::Int,
                            prefix="",
                            feature_projector = nothing)

    learner_key = prefix == "" ? "learner" : join([prefix, "learner"], "_")
    if  parsed[learner_key] ∈ ["LSTD", "LSTDLearner", "lstd"]
        learner_type = LSTDLearner
        eta_str = prefix == "" ? "eta" : join([prefix, "eta"], "_")
        lambda_str = prefix == "" ? "lambda" : join([prefix, "lambda"], "_")
        try
            LSTDLearner(parsed[eta_str],
                        parsed[lambda_str],
                        feature_size,
                        num_actions,
                        num_demons)
        catch
            throw("LSTDLearner needs: $(eta_str) (float), $(lambda_str) (float)")
        end
    else
        opt = get_optimizer(parsed, prefix)
        lu = get_learning_update(parsed, opt, prefix)

        learner_str = parsed[learner_key]
        demon_learner = if learner_str ∈ ["Q", "QLearner", "q"]
            LinearQLearner(lu,
                           feature_size,
                           num_actions,
                           num_demons)
        elseif learner_str ∈ ["SR", "SRLearner", "sr"]
            SRLearner(lu,
                      feature_size,
                      num_demons,
                      num_actions,
                      num_tasks,
                      feature_projector)
        elseif learner_str ∈ ["GPI", "gpi"]
            GPI(lu,
                feature_size,
                num_demons,
                num_actions,
                num_tasks,
                feature_projector)
        else
            throw(ArgumentError("Not a valid demon learner"))
        end
    end
end

function get_linear_learner(parsed::Dict,
                            feature_size,
                            num_actions,
                            demons,
                            prefix="",
                            feature_projector=nothing)

    num_demons = if demons isa Nothing
        1
    else
        length(demons)
    end

    learner_key = prefix == "" ? "learner" : join([prefix, "learner"], "_")
    if  parsed[learner_key] ∈ ["LSTD", "LSTDLearner", "lstd"]
        learner_type = LSTDLearner
        eta_str = prefix == "" ? "eta" : join([prefix, "eta"], "_")
        lambda_str = prefix == "" ? "lambda" : join([prefix, "lambda"], "_")
        try
            LSTDLearner(parsed[eta_str],
                        parsed[lambda_str],
                        feature_size,
                        num_actions,
                        num_demons)
        catch
            throw("LSTDLearner needs: $(eta_str) (float), $(lambda_str) (float)")
        end
    else
        opt = get_optimizer(parsed, prefix)
        lu = get_learning_update(parsed, opt, prefix)

        learner_str = parsed[learner_key]
        demon_learner = if learner_str ∈ ["Q", "QLearner", "q"]
            LinearQLearner(lu,
                           feature_size,
                           num_actions,
                           num_demons)
        elseif learner_str ∈ ["SR", "SRLearner", "sr"]
            SRLearner(lu,
                      feature_size,
                      num_demons,
                      num_actions,
                      demons.num_tasks,
                      feature_projector)
        elseif learner_str ∈ ["GPI", "gpi"]
            GPI(lu,
                feature_size,
                num_demons,
                num_actions,
                demons.num_tasks,
                feature_projector)
        else
            throw(ArgumentError("Not a valid demon learner"))
        end
    end
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

function _init_learning_update(lu_type::Union{Type{TB}, Type{TBTrueOnline}},
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
