

struct ReplacingTraces <: AbstractTraceUpdate end
struct AccumulatingTraces <: AbstractTraceUpdate end

_accumulate_trace(::AccumulatingTraces, e::AbstractMatrix, s::AbstractVector, pred_inds::Nothing) =
    e .+= s'
_accumulate_trace(::AccumulatingTraces, e::AbstractMatrix, s::AbstractVector, pred_inds) =
    e[pred_inds,:] .+= s'

_accumulate_trace(::AccumulatingTraces, e::AbstractMatrix, s::AbstractMatrix, pred_inds::Nothing) =
    e .+= s
_accumulate_trace(::AccumulatingTraces, e::AbstractMatrix, s::AbstractMatrix, pred_inds) =
    e[pred_inds,:] .+= s

_accumulate_trace(::ReplacingTraces, e::AbstractMatrix, s::AbstractVector, pred_inds::Nothing) =
    e .= s'
_accumulate_trace(::ReplacingTraces, e::AbstractMatrix, s::AbstractVector, pred_inds) =
    e[pred_inds,:] .= s'

_accumulate_trace(::ReplacingTraces, e::AbstractMatrix, s::AbstractMatrix, pred_inds::Nothing) =
    e .= s
_accumulate_trace(::ReplacingTraces, e::AbstractMatrix, s::AbstractMatrix, pred_inds) =
    e[pred_inds,:] .= s

function update_trace!(t::AbstractTraceUpdate, e::AbstractMatrix, s, λ, discounts, ρ, pred_inds=nothing)
    e .*= λ * discounts .* ρ
    _accumulate_trace(t, e, s, pred_inds)
end

function get_demon_pis(horde::GVFSRHordes.SRHorde, num_actions, state, obs)
    target_pis = zeros(length(horde), num_actions)
    for i in 1:length(horde.PredHorde)
        for a in 1:num_actions
            target_pis[i, a] = get(GVFHordes.policy(horde.PredHorde.gvfs[i]); state_t = obs, action_t = a)
        end
    end

    state_action_feature_length = Int( length(horde.SFHorde) / horde.num_SFs)
    inds_per_task = collect(1:state_action_feature_length:length(horde.SFHorde))

    for a in 1:num_actions
        unique_pi_SF = map(gvf -> get(GVFHordes.policy(gvf); state_t=obs, action_t=a), horde.SFHorde.gvfs[inds_per_task])
        upsf = repeat(unique_pi_SF, inner = state_action_feature_length)
        target_pis[length(horde.PredHorde)+1:end, a] .=  upsf
    end

    return target_pis
end

function get_demon_pis(horde, num_actions, state, obs)
    target_pis = zeros(length(horde), num_actions)
    for i in 1:length(horde)
        for a in 1:num_actions
            # target_pis[i, a] = get(GVFHordes.policy(horde.gvfs[i]), obs, a)
            target_pis[i, a] = get(GVFHordes.policy(horde.gvfs[i]); state_t = obs, action_t = a)
        end
    end
    return target_pis
end

function get_demon_parameters(lu::LearningUpdate, learner, demons, obs, state, action, next_obs, next_state, next_action, env_reward)
    C, next_discounts, _ = get(demons; state_t = obs, action_t = action, state_tp1 = next_obs, action_tp1 = next_action, reward = env_reward)
    target_pis = get_demon_pis(demons, learner.num_actions, state, obs)
    next_target_pis = get_demon_pis(demons, learner.num_actions, next_state, next_obs)
    C, next_discounts, target_pis, next_target_pis
end
