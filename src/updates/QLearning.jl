mutable struct QLearning <: Learner
    α::Float64
    γ::Float64
    num_actions::Int
    function QLearning(α, γ, num_actions)
        new(α, γ, num_actions)
    end
end


function update!(learner::QLearning,
                 weights::Array{Float64, 2},
                 s_t,
                 a_t,
                 s_tp1,
                 r_tp1,
                 terminal)

    α = learner.α
    γ = learner.γ

    weights*s_t

end

function zero_eligibility_traces!(learner::TB)
    learner.e .= 0
end
