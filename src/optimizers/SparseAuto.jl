import LinearAlgebra: dot
using Tullio, LoopVectorization

struct SparseAuto
    μ::Float64
    τ::Int
    M_Δ::Float64
    α_init::Float64
    α::IdDict
    h::IdDict
    n::IdDict
end

SparseAuto(μ, α_init, τ=10000, M_Δ=1.0) = SparseAuto(μ, τ, M_Δ, α_init, IdDict(), IdDict(), IdDict())



# Assumes ϕ and θ are the same size!!!!
function Flux.Optimise.update!(opt::SparseAuto, θ::AbstractArray{F}, ϕ, δ, z, num_demons, num_rows_per_demon) where {F<:AbstractFloat}
    α = get!(()->zero(θ) .+ F(opt.α_init), opt.α, θ)::typeof(θ)
    h = get!(()->zero(θ), opt.h, θ)::typeof(θ)
    n = get!(()->zero(θ) .+ 1, opt.n, θ)::typeof(θ)

    # @show length(findall(!iszero, α .- opt.α_init))
    # @show size(α)

    M_Δ, μ, τ = opt.M_Δ, opt.μ, opt.τ

    sp_ϕ = sparse(ϕ)
    
    δϕ = sparse_broadcast(*, δ, sp_ϕ)
    abs_ϕ = abs.(sp_ϕ)

    hδϕ = sparse_broadcast(*, δϕ, h)
    hdpmn = sparse_broadcast(-, abs.(hδϕ), n)
    sparse_broadcast!(+, n, (1/τ) * sparse_broadcast!(*, sparse_broadcast(*, abs_ϕ, hdpmn), α))

    nz_idx = findnz(sp_ϕ)
    ϕ_nz_idx = CartesianIndex.(nz_idx[1], nz_idx[2])
    # # ϕ_nz_idx = SparseArrays.nonzeroinds(sp_ϕ)
    
    n_ϕ_nz = @view n[ϕ_nz_idx]
    hδϕ_ϕ_nz = @view hδϕ[ϕ_nz_idx]
    α_ϕ_nz = @view α[ϕ_nz_idx]
    abs_ϕ_nz = @view abs_ϕ[ϕ_nz_idx]

    # # Δβ = sign.(hδϕ_ϕ_nz) .* min.(abs.(hδϕ_ϕ_nz./n_ϕ_nz), M_Δ)
    # # α_ϕ_nz .= min.(α_ϕ_nz .* exp.(μ * Δβ), 1.0 ./(abs_ϕ[ϕ_nz_idx]))
    @tullio Δβ[i] := clamp(hδϕ_ϕ_nz[i] / n_ϕ_nz[i], -1, 1)
    @tullio α_ϕ_nz[i] = clamp(α_ϕ_nz[i] * exp(μ * Δβ[i]), 1e-6, 1 / abs_ϕ_nz[i])

    @inbounds for d ∈ 1:num_demons
        rr = ((d-1)*num_rows_per_demon + 1):(d*num_rows_per_demon)
        α_d = @view α[rr, :]
        z_d = @view z[rr, :]
        if dot(α_d, z_d) > 1
            z_nz_idx = z_d .!= 0.0
            α_z_nz = @view α_d[z_nz_idx]
            α_z_nz .= min.(α_z_nz, 1 ./ sum(abs.(z_d)))
        end
    end

    sparse_broadcast!(+, θ, sparse_broadcast(*, δϕ, α))
    δ_h = sparse_broadcast(-, δϕ, sparse_broadcast(*, abs_ϕ, h))
    sparse_broadcast!(+, h, sparse_broadcast(*, δ_h, α))

end

function sparse_broadcast(m, x::AbstractSparseMatrix, y::AbstractMatrix)
    rows = rowvals(x)
    vals = nonzeros(x)
    m, n = size(x)
    ret = spzeros(promote_type(eltype(x), eltype(y)), size(x)...)
    for j = 1:n
        for i in nzrange(x, j)
            row = rows[i]
            val = vals[i]
            # perform sparse wizardry...
            ret[row, j] = val * y[row, j]
        end
    end
    ret
end

function sparse_broadcast!(m, x::AbstractSparseMatrix, y::AbstractMatrix)
    rows = rowvals(x)
    vals = nonzeros(x)
    m, n = size(x)
    for j = 1:n
        for i in nzrange(x, j)
            row = rows[i]
            val = vals[i]
            # perform sparse wizardry...
            x[row, j] = val * y[row, j]
        end
    end
    x
end

function sparse_broadcast!(m, x::AbstractMatrix, y::AbstractSparseMatrix)
    rows = rowvals(y)
    vals = nonzeros(y)
    m, n = size(y)
    for j = 1:n
        for i in nzrange(y, j)
            row = rows[i]
            val = vals[i]
            # perform sparse wizardry...
            x[row, j] = val * y[row, j]
        end
    end
    x
end

function sparse_broadcast(m, x::SparseMatrixCSC, y::AbstractVector)
    rows = rowvals(x)
    vals = nonzeros(x)
    m, n = size(x)
    ret = spzeros(promote_type(eltype(x), eltype(y)), size(x)...)
    for j = 1:n
        for i in nzrange(x, j)
            row = rows[i]
            val = vals[i]
            # perform sparse wizardry...
            ret[row, j] = val * y[row]
        end
    end
    ret
end

function sparse_broadcast(m, x::AbstractVector, y::SparseMatrixCSC)
    rows = rowvals(y)
    vals = nonzeros(y)
    m, n = size(y)
    ret = spzeros(promote_type(eltype(x), eltype(y)), size(y)...)
    # @show eltype(x), eltype(y), promote_rule(eltype(x), eltype(y))
    for j = 1:n
        for i in nzrange(y, j)
            row = rows[i]
            val = vals[i]
            # perform sparse wizardry...
            ret[row, j] = val * x[row]
        end
    end
    ret
end

