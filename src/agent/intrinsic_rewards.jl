# Test function taken from https://discourse.julialang.org/t/how-do-you-unfold-a-nested-julia-array/2243/3
# Helps flatten nested arrays
function flattenall(a)
    while any(x -> typeof(x) <: AbstractArray, a)
        a = collect(Iterators.flatten(a))
    end
    return a
end

mutable struct WeightChange{L<:Learner, A} <: IntrinsicReward
    learner::L
    previous_weights::A
    previous_reward::Float64
end

function WeightChange(learner::Learner)
    current_ws = flattenall(get_weights(learner))
    WeightChange(learner, deepcopy(current_ws), 0.0)
end

function update_reward!(wc::WeightChange, agent)
    cw = flattenall(get_weights(agent.demon_learner))
    curiosity_reward = sum(abs.(cw .- wc.previous_weights))
    wc.previous_weights .= deepcopy(cw)
    wc.previous_reward = curiosity_reward
    return curiosity_reward
end

function get_reward(wc::WeightChange, agent)
    return wc.previous_reward
end

mutable struct NoReward <: IntrinsicReward end

function update_reward!(::NoReward, agent)
    return 0
end

function get_reward(NR::NoReward, agent)
    return 0.0
end
