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

(e::EpsilonGreedy)(qs) = get_action_probs(e, qs)
(e::EpsilonGreedy)(qs, action) = get_action_probs(e, qs)[action]
