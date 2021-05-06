
abstract type LearningUpdate end
abstract type Learner end

export QLearner, LinearQLearner, SRLearner, LSTDLearner, GPI, predict
include("learners/value.jl")
include("learners/SR.jl")
include("learners/GPI.jl")
include("learners/LSTD.jl")

include("LearnedPolicy.jl")

update!(learner::Learner, args...) =
    update!(update(learner), learner, args...)

zero_eligibility_traces!(l::Learner) = zero_eligibility_traces!(l.update)


export TB, TBAuto, ESARSA, SR, update!, SARSA
abstract type AbstractTraceUpdate end
export AccumulatingTraces, ReplacingTraces
include("updates/update_utils.jl")
include("updates/TB.jl")
include("updates/SARSA.jl")
# include("updates/TB_Auto.jl")
include("updates/ESARSA.jl")
# include("updates/SR.jl")
include("updates/EmphESARSA.jl")
include("updates/ETB.jl")
include("updates/PriorESARSA.jl")
include("updates/PriorTB.jl")

