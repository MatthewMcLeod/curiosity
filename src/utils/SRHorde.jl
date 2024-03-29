module GVFSRHordes
import Lazy: @forward

export SFHorde, SRHorde
using GVFHordes
using ..Curiosity: FeatureCreator

struct SFHorde{T<:AbstractGVF} <: GVFHordes.AbstractHorde
    gvfs::Vector{T}
end

# mutable struct SRHorde{T<:GVFHordes.AbstractHorde} <: GVFHordes.AbstractHorde
mutable struct SRHorde <: GVFHordes.AbstractHorde
    PredHorde::GVFHordes.AbstractHorde
    SFHorde::SFHorde
    num_tasks::Int
    num_SFs::Int
    # state_constructor::Union{AbstractFeatureProjector, Curiosity.TileCoder}
    state_constructor::FeatureCreator
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
    # The reward discounts should always be 0 as they used for supervised learning prediction.
    @assert sum(discounts) == 0
    # For efficiency, the discount and pi for a given SF task should all have the same values. Therefore, to
    # improve speed, a single call to an GVF within each block should be sufficient.
    state_action_feature_length = Int( length(gvfh.SFHorde) / gvfh.num_SFs)
    inds_per_task = collect(1:state_action_feature_length:length(gvfh.SFHorde))

    constructed_state = gvfh.state_constructor(kwargs[:state_t], kwargs[:action_t], kwargs[:state_tp1])

    # @show constructed_state, kwargs[:action_t]
    #NOTE: Commenting out potential optimization as it is hacky and not too big of saving.
    # nz_inds = findall(!iszero, constructed_state)
    # num_actions = 4
    # C_SF_nz_ind = (nz_inds .- 1) * num_actions .+ kwargs[:action_t]
    # C_SF_per_SF = zeros(state_action_feature_length)
    # C_SF_per_SF[C_SF_nz_ind] .= 1
    # C_SF_fast = repeat(C_SF_per_SF, gvfh.num_SFs)
    #
    # cumulant_inds_per_task = state_action_feature_length
    # # C_SF = C_SF_fast
    C_SF = map(gvf -> get(cumulant(gvf); constructed_state_t = constructed_state, kwargs...), gvfh.SFHorde.gvfs)
    # #
    # if C_SF != C_SF_fast
    #     throw(ArgumentError("Fast version is not the same"))
    #     @show size(C_SF), size(C_SF_fast)
    # end
    # C_SF = zeros(length(gvfh.SFHorde))
    #
    # @show findall(!iszero, C_SF)
    # println()
    # println()

    # unique_discounts_SF = map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.SFHorde.gvfs[inds_per_task])
    unique_discounts_SF = map(gvf -> get(discount(gvf); kwargs...), gvfh.SFHorde.gvfs[inds_per_task])
    # unique_pi_SF = map(gvf -> get(policy(gvf), state_t, action_t), gvfh.SFHorde.gvfs[inds_per_task])
    unique_pi_SF = map(gvf -> get(policy(gvf); kwargs...), gvfh.SFHorde.gvfs[inds_per_task])

    discounts_SF = repeat(unique_discounts_SF, inner = state_action_feature_length)
    pi_SF = repeat(unique_pi_SF, inner = state_action_feature_length)

    return vcat(C,C_SF), vcat(discounts,discounts_SF), vcat(pi,pi_SF)
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
    preds_tp1 = nothing)

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
    preds_tp1 = nothing)

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
