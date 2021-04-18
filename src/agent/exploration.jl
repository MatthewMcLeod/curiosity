using Random

mutable struct EpsilonGreedy <: ExplorationStrategy
    epsilon::Number
end

function get_action_probs(EG::EpsilonGreedy, qs)
    m = maximum(qs)
    probs = zeros(length(qs))

    maxes_ind = findall(x-> x==m, qs)
    for ind in maxes_ind
        probs[ind] += (1 - EG.epsilon) / size(maxes_ind)[1]
    end
    probs .+= EG.epsilon/ length(qs)
    return probs
end

function step!(EG::EpsilonGreedy)
end

(e::EpsilonGreedy)(qs) = get_action_probs(e, qs)
(e::EpsilonGreedy)(qs, action) = get_action_probs(e, qs)[action]



"""
    Taken from Matt Schlegel Action RNN repo:
    https://github.com/mkschleg/ActionRNNs.jl/blob/master/src/ActingPolicy.jl
    Commit Hash: 5fa7dd8ace97014d7240d1e857c113d6f0c00e4e

    ϵGreedyDecay{AS}(ϵ_range, decay_period, warmup_steps, action_set::AS)
    ϵGreedyDecay(ϵ_range, end_step, num_actions)
This is an acting policy which decays exploration linearly over time. This api will possibly change overtime once I figure out a better way to specify decaying epsilon.
# Arguments
`ϵ_range::Tuple{Float64, Float64}`: (max epsilon, min epsilon)
`decay_period::Int`: period epsilon decays
`warmup_steps::Int`: number of steps before decay starts
"""
Base.@kwdef mutable struct ϵGreedyDecay{AS} <: ExplorationStrategy
    ϵ_range::Tuple{Float64, Float64}
    decay_period::Int
    warmup_steps::Int
    cur_step::Int = 0
    action_set::AS
    ϵGreedyDecay(ϵ_range, decay_period, warmup_steps, action_set::AS) where {AS} =
        new{AS}(ϵ_range, decay_period, warmup_steps, 0, action_set)
end

ϵGreedyDecay(ϵ_range, end_step, num_actions) = ϵGreedyDecay(ϵ_range, end_step, 1:num_actions)

action_set(ap::ϵGreedyDecay) = ap.action_set

function _get_eps_for_step(ap::ϵGreedyDecay, step=ap.cur_step)
    ϵ_min = ap.ϵ_range[2]
    ϵ_max = ap.ϵ_range[1]

    steps_left = ap.decay_period + ap.warmup_steps - ap.cur_step
    bonus = (ϵ_max - ϵ_min) * steps_left / ap.decay_period
    bonus = clamp(bonus, 0.0, ϵ_max - ϵ_min)
    ϵ_min + bonus
end

function get_action_probs(ap::ϵGreedyDecay, qs, step=ap.cur_step)
    ϵ = _get_eps_for_step(ap, step)

    m = maximum(qs)
    probs = zeros(length(qs))

    maxes_ind = findall(x-> x==m, qs)
    for ind in maxes_ind
        probs[ind] += (1 - ϵ) / size(maxes_ind)[1]
    end
    probs .+= ϵ/ length(qs)
    return probs
end

function step!(ap::ϵGreedyDecay)
    ap.cur_step += 1
end
(e::ϵGreedyDecay)(qs) = get_action_probs(e, qs)
(e::ϵGreedyDecay)(qs, action) = get_action_probs(e, qs)[action]
