module MountainCarUtils
using GVFHordes
# import StatsBase
using StatsBase
import Random

const discount = 0.99
import ..MountainCarConst
import ..GVFSRHordes


function create_demons(parsed, fc)
    action_space = 3
    demons = if parsed["demon_learner"] != "SR"
        GVFHordes.Horde(
            [steps_to_wall_gvf(),steps_to_goal_gvf()])
    elseif parsed["demon_learner"] == "SR"
        @assert demon_projected_fc != nothing
        pred_horde =  GVFHordes.Horde(
                [steps_to_wall_gvf(),steps_to_goal_gvf()])

        SF_policies = [g.policy for g in pred_horde.gvfs]
        SF_discounts = [g.discount for g in pred_horde.gvfs]
        num_SFs = length(SF_policies)
        SF_horde = SRCU.create_SF_horde(SF_policies, SF_discounts, demon_projected_fc,1:action_space)

        GVFSRHordes.SRHorde(pred_horde, SF_horde, num_SFs, demon_projected_fc)
    else
        throw(ArgumentError("Cannot create demons"))
    end
    return demons
end

struct MountainCarStateActionCumulant <: GVFParamFuncs.AbstractCumulant
    state_num::Int
    action::Int
    state_constructor::Any
end

function Base.get(cumulant::MountainCarStateActionCumulant; kwargs...)
    state = kwargs[:constructed_state_t]
    action = kwargs[:action_t]
    if state[cumulant.state_num] == 1 && action == cumulant.action
        return 1
    else
        return 0
    end
end

# function make_behaviour_gvf(discount, state_constructor_func, learner, exploration_strategy)
function make_behaviour_gvf(behaviour_learner, γ, fc, exploration_strategy)
    function b_π(state_constructor, learner, exploration_strategy; kwargs...)
        s = state_constructor(kwargs[:state_t])
        preds = learner(s)
        return exploration_strategy(preds)[kwargs[:action_t]]
    end
    GVF_policy = GVFParamFuncs.FunctionalPolicy((;kwargs...) -> b_π(fc, behaviour_learner, exploration_strategy; kwargs...))
    BehaviourGVF = GVF(GVFParamFuncs.RewardCumulant(), GVFParamFuncs.StateTerminationDiscount(γ, steps_to_goal_pseudoterm), GVF_policy)
end

function make_SF_for_policy(gvf_policy, gvf_pseudoterm, num_features, num_actions, state_constructor)
    return GVFSRHordes.SFHorde([GVF(MountainCarStateActionCumulant(s,a,state_constructor),
                    gvf_pseudoterm,
                    gvf_policy) for s in 1:num_features for a in 1:num_actions])
end

function make_SF_horde(discount, num_features, num_actions, state_constructor)
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

# NOTE IS THIS TP1 or just T
Base.get(π::EnergyPumpPolicy; kwargs...) =
    if kwargs[:state_t][2] >= vel_0(π)
        MountainCarConst.Accelerate == kwargs[:action_t] ? 1.0 : 0.0 # Go Right/Accelerate
    else
        MountainCarConst.Reverse == kwargs[:action_t] ? 1.0 : 0.0 # Go left/Reverse
    end

StatsBase.sample(rng::Random.AbstractRNG, π::EnergyPumpPolicy; kwargs...) =
    if kwargs[:state_t] state_t[2] >= vel_0(π)
        MountainCarConst.Accelerate # Go Right/Accelerate
    else
        MountainCarConst.Reverse # Go left/Reverse
    end
StatsBase.sample(rng::Random.AbstractRNG, π::EnergyPumpPolicy, state_t, actions) =
    sample(rng, Weights([get(π; state_t = state_t, action_t=a) for a in actions]))
StatsBase.sample(π::EnergyPumpPolicy, args...) =
    StatsBase.sample(Random.GLOBAL_RNG, π::EnergyPumpPolicy, args...)

# StatsBase.sample(rng::Random.AbstractRNG, π::EnergyPumpPolicy, state_t) =
#     sample(rng, Weights([get(π; state_t = state_t, action_t=a) for a in actions]))

function steps_to_wall_pseudoterm(;kwargs...)
    normed_wall_pos = 0.0
    # return obs[1] <= MountainCarConst.pos_limit[1]
    return kwargs[:state_tp1][1] <= normed_wall_pos
end
function steps_to_wall_cumulant(;kwargs...)
    normed_wall_pos = 0.0
    # return obs[1] <= MountainCarConst.pos_limit[1] ? 1.0 : 0.0
    return kwargs[:state_tp1][1] <= normed_wall_pos ? 1.0 : 0.0
end

function steps_to_goal_pseudoterm(;kwargs...)
    normed_goal_pos = 1.0
    # return obs[1] >= MountainCarConst.pos_limit[2]
    return kwargs[:state_tp1][1] >= normed_goal_pos
end

function steps_to_goal_cumulant(;kwargs...)
    normed_goal_pos = 1.0
    # return obs[1] >= MountainCarConst.pos_limit[2] ? 1.0 : 0.0
    return kwargs[:state_tp1][1] >= normed_goal_pos ? 1.0 : 0.0
end

function task_pseudoterm(;kwargs...)
    normed_goal_pos = 1.0
    return kwargs[:state_tp1][1] >= normed_goal_pos
end

function task_cumulant(;kwargs...)
    normed_goal_pos = 1.0
    # return obs[1] >= MountainCarConst.pos_limit[2] ? 1.0 : 0.0
    return kwargs[:state_tp1][1] >= normed_goal_pos ? 0.0 : -1.0
end

end #end MountainCarUtils
