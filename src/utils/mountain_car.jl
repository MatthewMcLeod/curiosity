module MountainCarUtils
using GVFHordes
import StatsBase
import Random

const discount = 0.99
import ..MountainCarConst
import ..GVFSRHordes

struct MountainCarStateActionCumulant <: GVFParamFuncs.AbstractCumulant
    state_num::Int
    action::Int
    state_constructor::Any
end

function Base.get(cumulant::MountainCarStateActionCumulant,obs,action,pred)
    # @show obs
    # state = cumulant.state_constructor(obs)
    state = obs
    # state = ones(4000)
    if state[cumulant.state_num] == 1 && action == cumulant.action
        # @show state
        return 1
    else
        return 0
    end
end

function make_SF_for_policy(gvf_policy, gvf_pseudoterm, num_features, num_actions, state_constructor)
    return GVFSRHordes.SFHorde([GVF(MountainCarStateActionCumulant(s,a,state_constructor),
                    gvf_pseudoterm,
                    gvf_policy) for s in 1:num_features for a in 1:num_actions])
end

function make_SF_horde(num_features, num_actions, state_constructor)
    steps_to_wall_horde = make_SF_for_policy(EnergyPumpPolicy(true),
                GVFParamFuncs.StateTerminationDiscount(discount, steps_to_wall_pseudoterm, 0.0),
                num_features, num_actions, state_constructor)

    steps_to_goal_horde = make_SF_for_policy(EnergyPumpPolicy(true),
                GVFParamFuncs.StateTerminationDiscount(discount, steps_to_goal_pseudoterm, 0.0),
                num_features, num_actions, state_constructor)

    SF_horde = GVFSRHordes.merge(steps_to_wall_horde,steps_to_goal_horde)
    return SF_horde
end



function MCNorm(obs)4
    pos_limit = MountainCarConst.pos_limit
    vel_limit = MountainCarConst.vel_limit
    return Float32[(obs[1] - pos_limit[1])/(pos_limit[2] - pos_limit[1]),
                   (obs[2] - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end

function task_gvf()
    GVF(GVFParamFuncs.FunctionalCumulant(task_cumulant),
        GVFParamFuncs.StateTerminationDiscount(0.95, task_pseudoterm, 0.0),
        EnergyPumpPolicy(true))
end

function steps_to_wall_gvf()
    GVF(GVFParamFuncs.FunctionalCumulant(steps_to_wall_cumulant),
        GVFParamFuncs.StateTerminationDiscount(discount, steps_to_wall_pseudoterm, 0.0),
        EnergyPumpPolicy(true))
end

function steps_to_goal_gvf()
    GVF(GVFParamFuncs.FunctionalCumulant(steps_to_goal_cumulant),
        GVFParamFuncs.StateTerminationDiscount(discount, steps_to_goal_pseudoterm, 0.0),
        EnergyPumpPolicy(true))
end

struct EnergyPumpPolicy <: GVFParamFuncs.AbstractPolicy
    normalized::Bool
    EnergyPumpPolicy(normalized=false) = new(normalized)
end

vel_0(π::EnergyPumpPolicy) = if π.normalized == true
    0.5
else
    0.0
end

Base.get(π::EnergyPumpPolicy, state_t, action_t) =
    if state_t[2] >= vel_0(π)
        MountainCarConst.Accelerate == action_t ? 1.0 : 0.0 # Go Right/Accelerate
    else
        MountainCarConst.Reverse == action_t ? 1.0 : 0.0 # Go left/Reverse
    end

StatsBase.sample(rng::Random.AbstractRNG, π::EnergyPumpPolicy, state_t) =
    if state_t[2] >= vel_0(π)
        MountainCarConst.Accelerate # Go Right/Accelerate
    else
        MountainCarConst.Reverse # Go left/Reverse
    end
StatsBase.sample(rng::Random.AbstractRNG, π::EnergyPumpPolicy, state_t, actions) =
    StatsBase.sample(rng::Random.AbstractRNG, π::EnergyPumpPolicy, state_t)
StatsBase.sample(π::EnergyPumpPolicy, args...) =
    StatsBase.sample(Random.GLOBAL_RNG, π::EnergyPumpPolicy, args...)

function steps_to_wall_pseudoterm(obs)
    normed_wall_pos = 0.0
    # return obs[1] <= MountainCarConst.pos_limit[1]
    return obs[1] <= normed_wall_pos
end
function steps_to_wall_cumulant(obs, action, pred)
    normed_wall_pos = 0.0
    # return obs[1] <= MountainCarConst.pos_limit[1] ? 1.0 : 0.0
    return obs[1] <= normed_wall_pos ? 1.0 : 0.0
end

function steps_to_goal_pseudoterm(obs)
    normed_goal_pos = 1.0
    # return obs[1] >= MountainCarConst.pos_limit[2]
    return obs[1] >= normed_goal_pos
end

function steps_to_goal_cumulant(obs, action, pred)
    normed_goal_pos = 1.0
    # return obs[1] >= MountainCarConst.pos_limit[2] ? 1.0 : 0.0
    return obs[1] >= normed_goal_pos ? 1.0 : 0.0
end

function task_pseudoterm(obs)
    normed_goal_pos = 1.0
    return obs[1] >= normed_goal_pos
end

function task_cumulant(obs, action, pred)
    normed_goal_pos = 1.0
    # return obs[1] >= MountainCarConst.pos_limit[2] ? 1.0 : 0.0
    return obs[1] >= normed_goal_pos ? 0.0 : -1.0
end

end #end MountainCarUtils
