# Test function taken from https://discourse.julialang.org/t/how-do-you-unfold-a-nested-julia-array/2243/3
# Helps flatten nested arrays
function flattenall(a::AbstractArray)
    while any(x -> typeof(x) <: AbstractArray, a)
        a = collect(Iterators.flatten(a))
    end
    return a
end

mutable struct WeightChange <: IntrinsicReward
    previous_weights::Array

    function WeightChange(weights)
        current_ws = deepcopy(weights)
        new(current_ws)
    end
end

function update_reward!(self::WeightChange, agent)
    current_ws = deepcopy(get_weights(agent.demon_learner))
    curiosity_reward = sum(abs.(flattenall(current_ws - self.previous_weights)))
    self.previous_weights = current_ws
    return curiosity_reward
end

mutable struct NoReward <: IntrinsicReward end

function update_reward!(::NoReward, agent)
    return 0
end
