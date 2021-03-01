module MountainCarUtils
using GVFHordes

const discount = 0.9
# const pos_limit = (-1.2, 0.5)
import ..MountainCarConst


function steps_to_wall_gvf()
    # GVF(GVFParamFuncs.FunctionalCumulant((args...)->return 0), GVFParamFuncs.ConstantDiscount(1.0), GVFParamFuncs.RandomPolicy(fill(1/3,3)))
    GVF(GVFParamFuncs.FunctionalCumulant(steps_to_wall_cumulant), GVFParamFuncs.StateTerminationDiscount(discount, steps_to_wall_pseudoterm, 0.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy) )
end

function steps_to_goal_gvf()
    # GVF(GVFParamFuncs.FunctionalCumulant((args...)->return 0), GVFParamFuncs.ConstantDiscount(1.0), GVFParamFuncs.RandomPolicy(fill(1/3,3)))
    # GVF(GVFParamFuncs.FunctionalCumulant((args...)->return 0), GVFParamFuncs.ConstantDiscount(1.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy))
    # GVF(GVFParamFuncs.FunctionalCumulant(steps_to_goal_cumulant), GVFParamFuncs.ConstantDiscount(1.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy))
    GVF(GVFParamFuncs.FunctionalCumulant(steps_to_goal_cumulant), GVFParamFuncs.StateTerminationDiscount(discount, steps_to_goal_pseudoterm, 0.0), GVFParamFuncs.FunctionalPolicy(energy_pump_policy))

end

function energy_pump_policy(obs, action)
    # Energy pumping is in the direction of the velocity
    action_to_take = if obs[2] >= 0
        3 # Go Right/Accelerate
    else
        1 # Go left/Reverse
    end
    return action_to_take == action ? 1.0 : 0.0
end

function steps_to_wall_pseudoterm(obs)
    return obs[1] >= MountainCarConst.pos_limit[1]
end

function steps_to_goal_pseudoterm(obs)
    return obs[1] >= MountainCarConst.pos_limit[2]
end

function steps_to_wall_cumulant(obs, action, pred)
    return obs[1] >= MountainCarConst.pos_limit[1] ? 1.0 : 0.0
end

function steps_to_goal_cumulant(obs, action, pred)
    return obs[1] >= MountainCarConst.pos_limit[2] ? 1.0 : 0.0
end


# Cumulant(gvf, s) = s == goal ? 1 : 0
# Discount(gvf, s) = s == goal ? 0.0 : \gamma


end #end MountainCarUtils
