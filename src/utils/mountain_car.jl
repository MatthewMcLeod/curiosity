module MountainCarUtils
using GVFHordes

function steps_to_wall_gvf()
    # GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions)))
    GVF(GVFParamFuncs.FunctionalCumulant((args...)->return 0), GVFParamFuncs.ConstantDiscount(1.0), GVFParamFuncs.RandomPolicy(fill(1/3,3)))
end

function steps_to_goal_gvf()
    # GVF(GVFParamFuncs.FeatureCumulant(i+1), GVFParamFuncs.StateTerminationDiscount(discount, pseudoterm), GVFParamFuncs.RandomPolicy(fill(1/num_actions,num_actions)))
    GVF(GVFParamFuncs.FunctionalCumulant((args...)->return 0), GVFParamFuncs.ConstantDiscount(1.0), GVFParamFuncs.RandomPolicy(fill(1/3,3)))

end

function steps_to_wall_pseudoterm(pos)
    return pos[1] >= MountainCarConst.pos_limit[1]
end

function steps_to_wall_cumulant()
    return pos[1] >= MountainCarConst.pos_limit[1]
end

function steps_to_goal_pseudoterm(pos)
    return pos[2] >= MountainCarConst.pos_limit[2]
end


# Cumulant(gvf, s) = s == goal ? 1 : 0
# Discount(gvf, s) = s == goal ? 0.0 : \gamma




end #end MountainCarUtils
