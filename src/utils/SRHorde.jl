module GVFSRHordes
import Lazy: @forward

export SRHorde

using GVFHordes

struct SRHorde{T<:AbstractGVF} <: GVFHordes.AbstractHorde
    gvfs::Vector{T}
end

merge(h1::SRHorde, h2::SRHorde) =
    SRHorde([deepcopy(h1.gvfs); deepcopy(h2.gvfs)])

function cumulant(gvf::AbstractGVF) end
function discount(gvf::AbstractGVF) end
function policy(gvf::AbstractGVF) end

cumulant(gvf::GVF) = gvf.cumulant
discount(gvf::GVF) = gvf.discount
policy(gvf::GVF) = gvf.policy



function Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C = map(gvf -> get(cumulant(gvf), state_t, action_t, preds_tp1), gvfh.gvfs)
    Γ = map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs = map(gvf -> get(policy(gvf), state_t, action_t), gvfh.gvfs)
    return C, Γ, Π_probs
end

function Base.get!(C::Array{T, 1}, Γ::Array{F, 1}, Π_probs::Array{H, 1}, gvfh::SRHorde, state_t, action_t, state_tp1, action_tp1, preds_tp1) where {T, F, H}
    C .= map(gvf -> get(cumulant(gvf), state_t, action_t, preds_t), gvfh.gvfs)
    Γ .= map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs .= map(gvf -> get(policy(gvf), state_t, action_t), gvfh.gvfs)
    return C, Γ, Π_probs
end

Base.get(gvfh::SRHorde, state_tp1, preds_tp1) =
    get(gvfh::SRHorde, nothing, nothing, state_tp1, nothing, preds_tp1)

Base.get(gvfh::SRHorde, state_t, action_t, state_tp1) =
    get(gvfh::SRHorde, state_t, action_t, state_tp1, nothing, nothing)

Base.get(gvfh::SRHorde, state_t, action_t, state_tp1, preds_tp1) =
    get(gvfh::SRHorde, state_t, action_t, state_tp1, nothing, preds_tp1)

@forward SRHorde.gvfs Base.length
end #end of module
