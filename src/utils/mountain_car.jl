module MountainCarUtils
using GVFHordes

const discount = 0.99
import ..MountainCarConst

function MCNorm(obs)
    pos_limit = MountainCarConst.pos_limit
    vel_limit = MountainCarConst.vel_limit
    return Float32[(obs[1] - pos_limit[1])/(pos_limit[2] - pos_limit[1]),
                   (obs[2] - vel_limit[1])/(vel_limit[2] - vel_limit[1])]
end

function task_gvf()
    GVF(GVFParamFuncs.FunctionalCumulant(task_cumulant), GVFParamFuncs.StateTerminationDiscount(0.9, task_pseudoterm, 0.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy) )
end

function steps_to_wall_gvf()
    GVF(GVFParamFuncs.FunctionalCumulant(steps_to_wall_cumulant), GVFParamFuncs.StateTerminationDiscount(discount, steps_to_wall_pseudoterm, 0.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy) )
end

function steps_to_goal_gvf()
    GVF(GVFParamFuncs.FunctionalCumulant(steps_to_goal_cumulant), GVFParamFuncs.StateTerminationDiscount(discount, steps_to_goal_pseudoterm, 0.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy))
end

function energy_pump_policy(obs, action)
    # Energy pumping is in the direction of the velocity
    #NOTE: Needs to know if the env is normalized. Always assume it is. Equivalent of 0 velocity is 0.5
    vel_0 = 0.5
    action_to_take = if obs[2] >= vel_0
        MountainCarConst.Accelerate # Go Right/Accelerate
    else
        MountainCarConst.Reverse # Go left/Reverse
    end
    return action_to_take == action ? 1.0 : 0.0
end

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
