module SRCreationUtils
    using GVFHordes
    import ..GVFSRHordes

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

struct TileCodeStateCumulant <: GVFParamFuncs.AbstractCumulant
    state_num::Int
end
function Base.get(cumulant::TileCodeStateCumulant; kwargs...)
    state = kwargs[:constructed_state_t]
    if state[cumulant.state_num] == 1
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

function create_SF_horde(policies, discounts, projected_feature_constructor, action_set)
    SF_horde = get_SF_horde_for_policy(policies[1], discounts[1], projected_feature_constructor, action_set)
    for i in 2:length(policies)
        SF_horde = GVFSRHordes.merge(SF_horde, get_SF_horde_for_policy(policies[i], discounts[i], projected_feature_constructor, action_set))
    end
    return SF_horde
end

function create_SF_for_policy_V2(policy,discount,reward_feature_constructor)
    num_features = size(reward_feature_constructor)
    SF = GVFSRHordes.SFHorde([GVF(
                        TileCodeStateCumulant(s),
                        discount,
                        policy) for s in 1:num_features])
    return SF

end

function create_SF_horde_V2(policies,discounts,projected_feature_constructor,action_set)
    SF_horde = create_SF_for_policy_V2(policies[1], discounts[1], projected_feature_constructor)
    for i in 2:length(policies)
        SF_horde = GVFSRHordes.merge(SF_horde, create_SF_for_policy_V2(policies[i], discounts[i], projected_feature_constructor))
    end
    return SF_horde
end

end
