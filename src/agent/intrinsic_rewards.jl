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
end

function WeightChange(learner::Learner)
    current_ws = flattenall(get_weights(learner))
    WeightChange(learner, current_ws)
end

function update_reward!(wc::WeightChange, agent)
    cw = flattenall(get_weights(agent.demon_learner))
    curiosity_reward = sum(abs.(cw .- wc.previous_weights))
    wc.previous_weights .= cw
    return curiosity_reward
end

mutable struct NoReward <: IntrinsicReward end

function update_reward!(::NoReward, agent)
    return 0
end
