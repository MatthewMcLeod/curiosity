module MountainCarLearnPolicy

using Curiosity, Statistics, Plots, FileIO, BSON
using Random, StatsBase, RollingFunctions
using GVFHordes, ProgressMeter

const MCU = Curiosity.MountainCarUtils

# Cumulant
struct Reward end

(c::Reward)(;kwargs...) = kwargs[:r]

struct ConstantDiscount{F}
    val::F
end

(d::ConstantDiscount)(;kwargs...) = d.val

# Environment
function construct_env()
    normalized = true
    env = MountainCar(0.0, 0.0, normalized)
end

# Agent
function construct_agent(numtilings, numtiles, lu_str, α, λ, ϵ, γ, policy_name)
    obs_size = 2
    fc = Curiosity.SparseTileCoder(numtilings, numtiles, obs_size)
    feature_size = size(fc)

    lu = if lu_str == "ESARSA"
        ESARSA(lambda=λ, opt=Curiosity.Descent(α), trace=ReplacingTraces())
    elseif lu_str == "SARSA"
        SARSA(lambda=λ, opt=Curiosity.Descent(α))
    elseif lu_str == "TB"
        TB(lambda=λ, opt=Curiosity.Descent(α))
    else
        throw(ArgumentError("$(lu_str) Not a valid behaviour learning update"))
    end
#     (update, num_features, num_actions, num_demons, w_init)
    w_init = 1 / numtilings
    learner = LinearQLearner(lu, feature_size, 3, 1,w_init)
    exploration = EpsilonGreedy(ϵ)
    # cumulant = Reward()
    # discount = ConstantDiscount(γ)
    # discount = GVFParamFuncs.ConstantDiscount(γ)

    cumulant,discount = if policy_name == "Goal"
        cumulant = MCU.step_to_goal_cumulant()
        discount = MCU.goal_pseudoterm(γ)
        cumulant,discount
    elseif policy_name == "Wall"
        cumulant = MCU.step_to_wall_cumulant()
        discount = MCU.wall_pseudoterm(γ)
        cumulant,discount
    elseif policy_name == "Task"
        cumulant = MCU.leave_cumulant()
        discount = MCU.goal_pseudoterm(γ)
        cumulant,discount
    else
        throw(ArgumentError("What Policy do you want to learn?? "))
    end

    b_gvf = make_behaviour_gvf(learner, discount, cumulant, fc, exploration)
    b_demons = Horde([b_gvf])

    Curiosity.PolicyLearner(learner,
                            fc,
                            exploration,
                            discount,
                            cumulant,
                            zeros(2),
                            0,
                            b_demons)
end

function make_behaviour_gvf(behaviour_learner, discount, cumulant, fc, exploration_strategy)
    function b_π(state_constructor, learner, exploration_strategy; kwargs...)
        s = state_constructor(kwargs[:state_t])
        preds = learner(s)
        return exploration_strategy(preds)[kwargs[:action_t]]
    end
    GVF_policy = GVFParamFuncs.FunctionalPolicy((;kwargs...) -> b_π(fc, behaviour_learner, exploration_strategy; kwargs...))
    BehaviourGVF = GVF(cumulant,discount, GVF_policy)
    return BehaviourGVF
end

function main_experiment(policy_name="tmp";progress=true, file_name = nothing)
    # Learn the policy

    if file_name isa Nothing
        file_name = policy_name
    end

    seed = 1029
    Random.seed!(seed)
    numtilings, numtiles = 8, 8
    lu_str = "ESARSA"
    α = 0.1/numtilings
    λ = 0.9
    ϵ = 0.0
    γ = 0.99

    info = Dict(
        "seed"=>seed,
        "numtilings"=>numtilings,
        "numtiles"=>numtiles,
        "lu"=>"lu_str",
        "α"=>α,
        "λ"=>λ,
        "ϵ"=>ϵ,
        "γ"=>γ,
        "rew"=>"Env"
    )


    env = construct_env()

    agent = construct_agent(numtilings, numtiles, lu_str, α, λ, ϵ, γ, policy_name)

    steps = Int[]
    ret = Float64[]
    max_num_steps = 300000
    eps = 0
    prg_bar = ProgressMeter.Progress(max_num_steps, "Step: ")

    while sum(steps) < max_num_steps
        is_terminal = false

        max_episode_steps = min(max_num_steps - sum(steps), 1000)
        s = start!(env)
        a = start!(agent, s)
        stp = 0
        a = 0
        tr = 0.0

        while !is_terminal && stp <= max_episode_steps
            s, r, is_terminal = MinimalRLCore.step!(env, a)
            a = MinimalRLCore.step!(agent, s, r, is_terminal)
            tr += r
            stp += 1

            if progress
                next!(prg_bar)
            end
        end
        push!(steps, stp)
        push!(ret, tr)

        eps += 1
    end
    Curiosity.save(agent, "./src/data/MC_learned_policies/$(policy_name).bson", info)

    steps,ret
end

end
