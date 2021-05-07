
using StatsBase
using LinearAlgebra


function get_freq(h, vals...)
    for (idx, val) ∈ enumerate(vals)
        if val < h.edges[idx][1] || val > h.edges[1][end]
            return eltype(h.weights)(0)
        end
    end
    proj = [searchsortedfirst(h.edges[i], val) for (i, val) ∈ enumerate(vals)]
    h.weights[proj...]
end

function get_freq(h, vals::AbstractVector)
    for (idx, val) ∈ enumerate(vals)
        if val < h.edges[idx][1] || val > h.edges[1][end]
            return eltype(h.weights)(0)
        end
    end
    proj = [searchsortedfirst(h.edges[i], val) for (i, val) ∈ enumerate(vals)]
    h.weights[proj...]
end

struct DPI{H<:Histogram, P}
    state_pdf::H
    policy::P
end

function DPI(states, policy; h_kwargs...)
    state_hist = fit(Histogram, (getindex.(states, i) for i in 1:size(states[1])); h_kwargs...)
    sh_norm = normalize(state_hist, mode=:pdf)
    DPI(state_hist, policy)
end

function (dpi::DPI)(state, action)
    s_p = get_freq(dpi.state_pdf, state)
    a_p = dpi.policy(state, action)
    s_p*a_p
end


