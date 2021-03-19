

abstract type ValueFunctionLearner <: Learner end

update(l::ValueFunctionLearner) = l.update

"""
   QLearner(model, num_actions, num_demons)
"""
mutable struct QLearner{F, LU<:LearningUpdate} <: ValueFunctionLearner
    model::F
    update::LU
    num_actions::Int
    num_demons::Int
end

LinearQLearner(update, num_features, num_actions, num_demons; init=(s...)->zeros(s...)) =
    QLearner(init(num_actions*num_demons, num_features), update, num_actions, num_demons)

predict(l::QLearner{F}, ϕ) where {F <: Matrix} = l.model*ϕ
predict(l::QLearner, ϕ) = l.model(ϕ)

(l::QLearner)(ϕ) = predict(l, ϕ)

predict(l::QLearner, ϕ::AbstractVector, a) = predict(l, ϕ)[a .+ (0:(l.num_demons-1)).*l.num_actions]
predict(l::QLearner, ϕ::AbstractMatrix, a) = predict(l, ϕ)[a .+ (0:(l.num_demons-1)).*l.num_actions, :]

(l::QLearner)(ϕ, a) = predict(l, ϕ, a)

is_linear(l::QLearner) = false
is_linear(l::QLearner{Matrix{<:Number}}) = true

<<<<<<< HEAD

# mutable struct VLearner{F, LU<:LearningUpdate} <: ValueFunctioner
=======
function get_weights(l::QLearner)
    return l.model
end
# mutable struct VLearner{F, LU<:LearningUpdate} <: ValueFunctionLearner
>>>>>>> 79a59e7ff935f1839f84618c7fe2500a90147634
#     model::F
#     update::LU
#     num_demons::Int
# end

# predict(l::VLearner{F}, ϕ) where {F <: Matrix} = l.model*ϕ
# predict(l::VLearner, ϕ) = l.model(ϕ)

# (l::VLearner)(ϕ) = predict(l.model, ϕ)

# is_linear(l::VLearner) = false
# is_linear(l::VLearner{Matrix{<:Number}}) = true
