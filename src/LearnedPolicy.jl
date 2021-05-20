
using GVFHordes
import BSON

struct LearnedPolicy{M, D} <: GVFHordes.GVFParamFuncs.AbstractPolicy
    policy::M
    info::D
end

function LearnedPolicy(loc::AbstractString)
    d = BSON.load(loc, @__MODULE__)
    if "LearnedPolicy" ∈ keys(d)
        d["LearnedPolicy"]
    elseif "learner" ∈ keys(d)
        LearnedPolicy(Flux.Chain(d["fc"], d["learner"]), d["info"])
    end
end

Base.get(π::LearnedPolicy, state_t, action_t) = π(state_t, action_t)

(lp::LearnedPolicy)(s) = lp.policy(s)
(lp::LearnedPolicy)(s, a) = lp.policy(s)[a]

# Policy learning struct
mutable struct PolicyLearner{L<:Learner, FC, E, D, C, Φ, DEMON} <: MinimalRLCore.AbstractAgent
    learner::L
    fc::FC
    exploration::E

    discount::D
    cumulant::C
    o_t::Φ
    a_t::Int
    BehaviourDemon::DEMON
end

function MinimalRLCore.start!(pl::PolicyLearner, o)
    s_t = pl.fc(o)
    pl.o_t = copy(o)
    pl.a_t = μ(pl, s_t)

    pl.a_t
end

function μ_dist(pl::PolicyLearner, state)
    qs = pl.learner(state)
    pl.exploration(qs)
end

function μ(pl::PolicyLearner, state)
    action_probs = μ_dist(pl, state)
    action = sample(1:length(action_probs), Weights(action_probs))
end

function MinimalRLCore.step!(pl::PolicyLearner, o_tp1, rew, term)
    s_tp1 = pl.fc(o_tp1)
    next_action = μ(pl, s_tp1)

    rew = pl.cumulant(;r=rew, o_t=pl.o_t, a_t=pl.a_t, o_tp1=o_tp1)
    # γ = if pl.discount isa Number
    #     pl.discount
    # else
    #     pl.discount(;r=rew, o_t=pl.o_t, a_t=pl.a_t, o_tp1=o_tp1)
    # end

    update!(pl.learner,
            pl.BehaviourDemon,
            pl.o_t,
            o_tp1,
            pl.fc(pl.o_t),
            pl.a_t,
            pl.fc(o_tp1),
            next_action,
            term,
            (state, obs) -> μ_dist(pl, state),
            rew)

    pl.o_t = copy(o_tp1)
    pl.a_t = next_action

    pl.a_t
end

save(pl::PolicyLearner, loc, info::AbstractDict) =
    BSON.bson(loc,
              Dict("learner"=>pl.learner,
                   "fc"=>pl.fc,
                   "exp"=>pl.exploration,
                   "info"=>info))
