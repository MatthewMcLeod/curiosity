
using StatsBase
using LinearAlgebra


function get_freq(h, vals...)
    for (idx, val) ∈ enumerate(vals)
        if val < h.edges[idx][1] || val >= h.edges[idx][end]
            return eltype(h.weights)(0)
        end
    end
    proj = [searchsortedlast(h.edges[i], val) for (i, val) ∈ enumerate(vals)]
    h.weights[proj...]
end

function get_freq(h, vals::AbstractVector)
    for (idx, val) ∈ enumerate(vals)
        if val < h.edges[idx][1] || val >= h.edges[idx][end]
            return eltype(h.weights)(0)
        end
    end
    proj = [searchsortedlast(h.edges[i], val) for (i, val) ∈ enumerate(vals)]
    h.weights[proj...]
end

struct DPI{H<:Histogram, P}
    state_pdf::H
    policy::P
end

function DPI(states, policy; h_kwargs...)
    state_hist = fit(Histogram, Tuple(getindex.(states, i) for i in 1:length(states[1])); h_kwargs...)
    sh_norm = normalize(state_hist, mode=:probability)
    DPI(sh_norm, policy)
end

function (dpi::DPI)(state, action)
    s_p = get_freq(dpi.state_pdf, state)
    a_p = get(dpi.policy, state_t=state, action_t=action)
    s_p*a_p
end

struct HordeDPI{F}
    dpis::Vector{DPI}
    state_filter::F
end

function (hdpi::HordeDPI)(state, action)
    [dpi(hdpi.state_filter(state),action) for dpi in hdpi.dpis]
end


