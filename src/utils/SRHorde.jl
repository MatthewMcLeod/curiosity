module GVFSRHordes
import Lazy: @forward

export SFHorde, SRHorde
using GVFHordes
using ..Curiosity: AbstractFeatureProjector

struct SFHorde{T<:AbstractGVF} <: GVFHordes.AbstractHorde
    gvfs::Vector{T}
end

# mutable struct SRHorde{T<:GVFHordes.AbstractHorde} <: GVFHordes.AbstractHorde
mutable struct SRHorde <: GVFHordes.AbstractHorde
    PredHorde::GVFHordes.AbstractHorde
    SFHorde::SFHorde
    num_tasks::Int
    num_SFs::Int
    state_constructor::AbstractFeatureProjector
    # state_constructor::Function
    function SRHorde(prediction_horde::GVFHordes.AbstractHorde, successor_feature_horde::SFHorde, num_SFs, state_constructor)
        new(prediction_horde, successor_feature_horde, length(prediction_horde), num_SFs, state_constructor)
    end
end

function assign_prediction_horde!(h::SRHorde, new_pred_horde)
    SRHorde.PredHorde = new_pred_horde
end

function Base.length(h::SRHorde)
    return length(h.PredHorde) + length(h.SFHorde)
end

# function Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
function Base.get(gvfh::SRHorde; kwargs...)
    C, discounts, pi = get(gvfh.PredHorde; kwargs...)

    constructed_state = gvfh.state_constructor(kwargs[:state_t])
    # C_SF = map(gvf -> get(cumulant(gvf), constructed_state, action_t, preds_tp1), gvfh.SFHorde.gvfs)
    C_SF = map(gvf -> get(cumulant(gvf);constructed_state_t = constructed_state, kwargs...), gvfh.SFHorde.gvfs)
    # For efficiency, the discount and pi for a given SF task should all have the same values. Therefore, to
    # improve speed, a single call to an GVF within each block should be sufficient.
    state_action_feature_length = Int( length(gvfh.SFHorde) / gvfh.num_SFs)
    inds_per_task = collect(1:state_action_feature_length:length(gvfh.SFHorde))
    # unique_discounts_SF = map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.SFHorde.gvfs[inds_per_task])
    unique_discounts_SF = map(gvf -> get(discount(gvf); kwargs...), gvfh.SFHorde.gvfs[inds_per_task])
    # unique_pi_SF = map(gvf -> get(policy(gvf), state_t, action_t), gvfh.SFHorde.gvfs[inds_per_task])
    unique_pi_SF = map(gvf -> get(policy(gvf); kwargs...), gvfh.SFHorde.gvfs[inds_per_task])

    discounts_SF = repeat(unique_discounts_SF, inner = state_action_feature_length)
    pi_SF = repeat(unique_pi_SF, inner = state_action_feature_length)

    return vcat(C,C_SF),vcat(discounts,discounts_SF), vcat(pi,pi_SF)
end

# Base.get(gvfh::SRHorde, state_tp1, preds_tp1) =
#     get(gvfh::SRHorde, nothing, nothing, state_tp1, nothing, preds_tp1)
#
# Base.get(gvfh::SRHorde, state_t, action_t, state_tp1) =
#     get(gvfh::SRHorde, state_t, action_t, state_tp1, nothing, nothing)
#
# Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, preds_tp1) =
#     get(gvfh::SRHorde, state_t, action_t, state_tp1, nothing, preds_tp1)

Base.get(gvfh::SRHorde, state_tp1, preds_tp1) =
    get(gvfh::SRHorde; state_t = nothing,
    action_t = nothing,
    state_tp1 = state_tp1,
    action_tp1 = nothing,
    preds_tp1)

Base.get(gvfh::SRHorde, state_t, action_t, state_tp1) =
    get(gvfh::SRHorde;
    state_t = state_t,
    action_t = action_t,
    state_tp1 = state_tp1,
    action_tp1 = nothing,
    preds_tp1 = nothing)

Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, preds_tp1) =
    get(gvfh::SRHorde;
    state_t = state_t,
    action_t = action_t,
    state_tp1 = state_tp1,
    action_tp1 = nothing,
    preds_tp1 = preds_tp1)

merge(h1::SFHorde, h2::SFHorde) =
    SFHorde([deepcopy(h1.gvfs); deepcopy(h2.gvfs)])

function cumulant(gvf::AbstractGVF) end
function discount(gvf::AbstractGVF) end
function policy(gvf::AbstractGVF) end

cumulant(gvf::GVF) = gvf.cumulant
discount(gvf::GVF) = gvf.discount
policy(gvf::GVF) = gvf.policy



function Base.get(gvfh::SFHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C = map(gvf -> get(cumulant(gvf), state_t, action_t, preds_tp1), gvfh.gvfs)
    Γ = map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs = map(gvf -> get(policy(gvf), state_t, action_t), gvfh.gvfs)
    return C, Γ, Π_probs
end

function Base.get!(C::Array{T, 1}, Γ::Array{F, 1}, Π_probs::Array{H, 1}, gvfh::SFHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1) where {T, F, H}
    C .= map(gvf -> get(cumulant(gvf), state_t, action_t, preds_t), gvfh.gvfs)
    Γ .= map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs .= map(gvf -> get(policy(gvf), state_t, action_t), gvfh.gvfs)
    return C, Γ, Π_probs
end

Base.get(gvfh::SFHorde, state_tp1, preds_tp1) =
    get(gvfh::SFHorde; state_t = nothing,
    action_t = nothing,
    state_tp1 = state_tp1,
    action_tp1 = nothing,
    preds_tp1)

Base.get(gvfh::SFHorde, state_t, action_t, state_tp1) =
    get(gvfh::SFHorde;
    state_t = state_t,
    action_t = action_t,
    state_tp1 = state_tp1,
    action_tp1 = nothing,
    preds_tp1 = nothing)

Base.get(gvfh::SFHorde, state_t, action_t, state_tp1, preds_tp1) =
    get(gvfh::SFHorde;
    state_t = state_t,
    action_t = action_t,
    state_tp1 = state_tp1,
    action_tp1 = nothing,
    preds_tp1 = preds_tp1)
#
# Base.get(gvfh::SFHorde, state_tp1, preds_tp1) =
#     get(gvfh::SFHorde, nothing, nothing, state_tp1, nothing, preds_tp1)
#
# Base.get(gvfh::SFHorde, state_t, action_t, state_tp1) =
#     get(gvfh::SFHorde, state_t, action_t, state_tp1, nothing, nothing)
#
# Base.get(gvfh::SFHorde, state_t, action_t, state_tp1, preds_tp1) =
#     get(gvfh::SFHorde, state_t, action_t, state_tp1, nothing, preds_tp1)

@forward SFHorde.gvfs Base.length
end #end of module
