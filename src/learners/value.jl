

abstract type ValueFunctionLearner <: Learner end

"""
   QLearner(model, num_actions, num_demons) 
"""
mutable struct QLearner{F} <: ValueFunctionLearner
    model::F
    num_actions::Int
    num_demons::Int
end

LinearQLearner(num_features, num_actions, num_demons; init=(s...)->zeros(s...)) =
    init(num_actions*num_demons, num_features)

predict(l::QLearner{F}, ϕ) where {F <: Matrix} = l.model*ϕ
predict(l::QLearner, ϕ) = l.model(ϕ)

(l::QLearner)(ϕ) = predict(l.model, ϕ)

predict(l::QLearner{F}, ϕ::Vector, a) = predict(l, ϕ)[a .+ (0:(l.num_demons-1)).*l.num_actions]
predict(l::QLearner{F}, ϕ::Matrix, a) = predict(l, ϕ)[a .+ (0:(l.num_demons-1)).*l.num_actions, :]

(l::QLearner)(ϕ, a) = predict(l.model, ϕ, a)

is_linear(l::QLearner) = false
is_linear(l::QLearner{Matrix{<:Number}}) = true


mutable struct VLearner{F} <: ValueFunctionLearner
    model::F
    num_demons::Int
end

predict(l::VLearner{F}, ϕ) where {F <: Matrix} = l.model*ϕ
predict(l::VLearner, ϕ) = l.model(ϕ)

(l::VLearner, ϕ) = predict(l.model, ϕ)

is_linear(l::VLearner) = false
is_linear(l::VLearner{Matrix{<:Number}}) = true

