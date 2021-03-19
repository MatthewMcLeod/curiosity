
abstract type LearningUpdate end
abstract type Learner end

export QLearner, LinearQLearner, SRLearner, LSTDLearner, predict
include("learners/value.jl")
include("learners/SR.jl")
include("learners/LSTD.jl")

update!(learner::Learner, args...) =
    update!(update(learner), learner, args...)

zero_eligibility_traces!(l::Learner) = zero_eligibility_traces!(l.update)

export TB, TBAuto, ESARSA, SR, update!

include("updates/update_utils.jl")
include("updates/TB.jl")
# include("updates/TB_Auto.jl")
include("updates/ESARSA.jl")
# include("updates/SR.jl")




