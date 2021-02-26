struct Auto
    μ::Float64
    τ::Float64
    M_Δ::Float64
    α_init::Float64
    α::IdDict
    h::IdDict
    n::IdDict
end


# Assumes ϕ and θ are the same size!!!!
function apply!(opt::Auto, θ::AbstractArray{F}, ϕ, δ, z) where {F<:AbstractFloat}
    α = get!(opt.α, θ, zero(θ) .+ F(opt.α_init))::typeof(θ)
    h = get!(opt.h, θ, zero(θ))::typeof(θ)
    n = get!(opt.n, θ, zero(θ))::typeof(θ)

    M_Δ, μ, τ = opt.M_Δ, opt.μ, opt.τ

    δϕ = δ * ϕ
    abs_ϕ = abs.(ϕ)
    hδϕ = (δϕ).*h
    n .+= (1/τ) * α.*abs_ϕ.*(abs.(hδϕ) - n)

    ϕ_nz_idx = ϕ .!= 0.0
    n_ϕ_nz = @view n[ϕ_nz_idx]
    hδϕ_ϕ_nz = @view hδϕ[ϕ_nz_idx]
    α_ϕ_nz = @view α[ϕ_nz_idx]

    Δβ = sign.(hδϕ_ϕ_nz) .* min.(abs.(hδϕ_ϕ_nz./n_ϕ_nz), M_Δ)
    α_ϕ_nz .= min.(α_ϕ_nz .* exp.(μ * Δβ), 1.0 ./(abs_ϕ[ϕ_nz]))

    if dot(α, z) > 1
        z_nz_idx = z .!= 0.0
        α_z_nz = @view α[z_nz_idx]
        α_z_nz .= min.(α_z_nz, 1.0 ./abs(@view z[z_nz_idx]))
    end

    θ .+= α.*δϕ
    h .= h.*(1 .- α.*abs_ϕ) + α.*δϕ
end
