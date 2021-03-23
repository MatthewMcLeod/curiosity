
abstract type AbstractTraceUpdate end

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
    for a in 1:num_actions
        _, _, pi = get(horde, obs, a, obs, a)
        target_pis[:,a] = pi
    end
    return target_pis
end

function get_demon_pis(horde, num_actions, state, obs)
    target_pis = zeros(length(horde), num_actions)
    for i in 1:length(horde)
        for a in 1:num_actions
            target_pis[i, a] = get(GVFHordes.policy(horde.gvfs[i]), obs, a)
        end
    end
    return target_pis
end
