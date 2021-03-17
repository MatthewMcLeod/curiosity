module GVFSRHordes
import Lazy: @forward

export SFHorde, SRHorde
using GVFHordes

struct SFHorde{T<:AbstractGVF} <: GVFHordes.AbstractHorde
    gvfs::Vector{T}
end

# mutable struct SRHorde{T<:GVFHordes.AbstractHorde} <: GVFHordes.AbstractHorde
mutable struct SRHorde <: GVFHordes.AbstractHorde
    PredHorde::GVFHordes.AbstractHorde
    SFHorde::SFHorde
    num_tasks::Int
    state_constructor::Function
    function SRHorde(prediction_horde::GVFHordes.AbstractHorde, successor_feature_horde::SFHorde, state_constructor)
        new(prediction_horde, successor_feature_horde, length(prediction_horde), state_constructor)
    end
end

function Base.length(h::SRHorde)
    return length(h.PredHorde) + length(h.SFHorde)
end

function Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C, discounts, pi = get(gvfh.PredHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    # NOTE: Assume discounts and pi align with the task definition in the Pred Horde.
    #Do not need to query the SF Horde to get this info

    constructed_state = gvfh.state_constructor(state_t)

    # C_SF = map(gvf -> get(cumulant(gvf), state_t, action_t, preds_tp1), gvfh.SFHorde.gvfs)
    C_SF = map(gvf -> get(cumulant(gvf), constructed_state, action_t, preds_tp1), gvfh.SFHorde.gvfs)
    # C_SF = ones(length(gvfh.SFHorde.gvfs))
    state_action_feature_length = Int( length(gvfh.SFHorde) / gvfh.num_tasks)
    discounts_SF = repeat(discounts, inner = state_action_feature_length)
    pi_SF = repeat(pi, inner = state_action_feature_length)
    return vcat(C,C_SF),vcat(discounts,discounts_SF), vcat(pi,pi_SF)
end

Base.get(gvfh::SRHorde, state_tp1, preds_tp1) =
    get(gvfh::SRHorde, nothing, nothing, state_tp1, nothing, preds_tp1)

Base.get(gvfh::SRHorde, state_t, action_t, state_tp1) =
    get(gvfh::SRHorde, state_t, action_t, state_tp1, nothing, nothing)

Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, preds_tp1) =
    get(gvfh::SRHorde, state_t, action_t, state_tp1, nothing, preds_tp1)

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
    get(gvfh::SFHorde, nothing, nothing, state_tp1, nothing, preds_tp1)

Base.get(gvfh::SFHorde, state_t, action_t, state_tp1) =
    get(gvfh::SFHorde, state_t, action_t, state_tp1, nothing, nothing)

Base.get(gvfh::SFHorde, state_t, action_t, state_tp1, preds_tp1) =
    get(gvfh::SFHorde, state_t, action_t, state_tp1, nothing, preds_tp1)

@forward SFHorde.gvfs Base.length
end #end of module
