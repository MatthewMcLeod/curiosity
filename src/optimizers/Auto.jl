import LinearAlgebra: dot


struct Auto
    μ::Float64
    τ::Int
    M_Δ::Float64
    α_init::Float64
    α::IdDict
    h::IdDict
    n::IdDict
end

Auto(μ, α_init, τ=10000, M_Δ=1.0) = Auto(μ, τ, M_Δ, α_init, IdDict(), IdDict(), IdDict())

# Assumes ϕ and θ are the same size!!!!
function Flux.Optimise.update!(opt::Auto, θ::AbstractArray{F}, ϕ, δ, z, num_demons, num_rows_per_demon) where {F<:AbstractFloat}
    α = get!(()->zero(θ) .+ F(opt.α_init), opt.α, θ)::typeof(θ)
    h = get!(()->zero(θ), opt.h, θ)::typeof(θ)
    n = get!(()->zero(θ) .+ 1, opt.n, θ)::typeof(θ)

    # @show length(findall(!iszero, α .- opt.α_init))
    # @show size(α)

    M_Δ, μ, τ = opt.M_Δ, opt.μ, opt.τ

    δϕ = δ .* ϕ
    # println(size(δϕ))
    abs_ϕ = abs.(ϕ)
    hδϕ = (δϕ).*h
    n .+= (1/τ) * α.*abs_ϕ.*(abs.(hδϕ) - n)

    ϕ_nz_idx = ϕ .!= 0.0
    n_ϕ_nz = @view n[ϕ_nz_idx]
    hδϕ_ϕ_nz = @view hδϕ[ϕ_nz_idx]
    α_ϕ_nz = @view α[ϕ_nz_idx]

    # Δβ = sign.(hδϕ_ϕ_nz) .* min.(abs.(hδϕ_ϕ_nz./n_ϕ_nz), M_Δ)
    # α_ϕ_nz .= min.(α_ϕ_nz .* exp.(μ * Δβ), 1.0 ./(abs_ϕ[ϕ_nz_idx]))
    Δβ = clamp.(hδϕ_ϕ_nz./n_ϕ_nz, -1, 1)
    α_ϕ_nz .= clamp.(α_ϕ_nz .* exp.(μ * Δβ), 1e-6, 1 ./ (abs_ϕ[ϕ_nz_idx]))

    for d ∈ 1:num_demons
        rr = ((d-1)*num_rows_per_demon + 1):(d*num_rows_per_demon)
        α_d = @view α[rr, :]
        z_d = @view z[rr, :]
        if dot(α_d, z_d) > 1
            z_nz_idx = z_d .!= 0.0
            α_z_nz = @view α_d[z_nz_idx]
            α_z_nz .= min.(α_z_nz, 1 ./sum(abs.(z_d)))
        # else
        #     # @show "Not Triggered for $(d)"
        end
    end
    θ .+= α.*δϕ
    # @show "after", sum(θ)
    h .= h.*(1 .- α.*abs_ϕ) + α.*δϕ


end
