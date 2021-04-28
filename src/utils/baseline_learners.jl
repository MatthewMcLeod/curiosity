module BaselineUtils

using Curiosity
using GVFHordes

import ..GVFSRHordes
import ..Learner

using Distributions

struct FollowDemon <: Learner
    demon::AbstractGVF
    action_set::UnitRange
    update::Any
    function FollowDemon(demon, action_set)
        new(demon, action_set,1)
    end
end

Curiosity.update!(learner::FollowDemon, args...) = 1

function get_action_probs(l::FollowDemon, state, obs, action)
    #NOTE: This is misleading... state_t for demons is observations
    return get(l.demon.policy, state_t = obs, action_t = action)
end

(l::FollowDemon)(obs) = [get_action_probs(l, nothing, obs, a) for a ∈ l.action_set]

struct RandomDemons <: Learner
    demons::GVFHordes.AbstractHorde
    action_set::UnitRange
    update::Any
    function RandomDemons(demons,action_set)
        new(demons,action_set,1)
    end
end


Curiosity.update!(learner::RandomDemons, args...) = 1

function get_action_probs(l::RandomDemons, state, obs, action)
    #NOTE: This is misleading... state_t for demons is observations
    π = map(gvf -> get(gvf.policy; state_t = obs, action_t = action), l.demons.gvfs)
    return sum(π) / length(l.demons)
end

(l::RandomDemons)(obs) = [get_action_probs(l, nothing, obs, a) for a ∈ l.action_set]


end
