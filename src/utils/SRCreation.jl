module SRCreationUtils
    using GVFHordes
    import ..GVFSRHordes
# function make_SF_for_policy(gvf_policy, gvf_pseudoterm, num_features, num_actions, state_constructor)
#     return GVFSRHordes.SFHorde([GVF(MountainCarStateActionCumulant(s,a,state_constructor),
#                     gvf_pseudoterm,
#                     gvf_policy) for s in 1:num_features for a in 1:num_actions])
# end
#
# function make_SF_horde(discount, num_features, num_actions, state_constructor)
#     steps_to_wall_horde = make_SF_for_policy(EnergyPumpPolicy(true),
#                 GVFParamFuncs.StateTerminationDiscount(discount, steps_to_wall_pseudoterm, 0.0),
#                 num_features, num_actions, state_constructor)
#
#     steps_to_goal_horde = make_SF_for_policy(EnergyPumpPolicy(true),
#                 GVFParamFuncs.StateTerminationDiscount(discount, steps_to_goal_pseudoterm, 0.0),
#                 num_features, num_actions, state_constructor)
#
#     SF_horde = GVFSRHordes.merge(steps_to_wall_horde,steps_to_goal_horde)
#     return SF_horde
# end

struct TileCodeStateActionCumulant <: GVFParamFuncs.AbstractCumulant
    state_num::Int
    action::Int
    state_constructor::Any
end

function Base.get(cumulant::TileCodeStateActionCumulant; kwargs...)
    state = kwargs[:constructed_state_t]
    action = kwargs[:action_t]
    if state[cumulant.state_num] == 1 && action == cumulant.action
        return 1
    else
        return 0
    end
end

```
Takes
```
function get_SF_horde_for_policy(policy, discount, projected_feature_constructor, action_set)
    num_features = size(projected_feature_constructor)
    SF = GVFSRHordes.SFHorde([GVF(
                        TileCodeStateActionCumulant(s,a,projected_feature_constructor),
                        discount,
                        policy) for s in 1:num_features for a in action_set])
    return SF

end


end
