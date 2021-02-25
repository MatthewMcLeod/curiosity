module Curiosity

abstract type Learner end

<<<<<<< HEAD
export TB, TBAuto, ESARSA, update!
=======
@reexport using MinimalRLCore


export Auto
include("optimizers/Auto.jl")

export TB, TBAuto, update!
>>>>>>> 1b53e536204cc23d03c734e0f3fccbc6199a85dc
include("learners/TB.jl")
include("learners/TB_Auto.jl")
include("learners/ESARSA.jl")

<<<<<<< HEAD
abstract type IntrinsicReward end
include("agent/intrinsic_rewards.jl")

=======
>>>>>>> 1b53e536204cc23d03c734e0f3fccbc6199a85dc
export TabularRoundRobin, update!
include("learners/TabularRoundRobin.jl")

export Agent, agent_end!, step!
include("agent/agent.jl")

abstract type CumulantSchedule end
function update! end
function get_cumulants end

<<<<<<< HEAD
export TabularTMaze, env_step!, env_start!, valid_state_mask
    # TabularMazeCumulantSchedules, get_cumulants, update!
include("environments/tabular_tmaze/tabular_tmaze.jl")
=======
export TabularTMaze, MountainCar, valid_state_mask
include("environments/tabular_tmaze.jl")
include("environments/mountain_car.jl")

# logger
export Logger, logger_step!
include("logger/logger.jl")
>>>>>>> 1b53e536204cc23d03c734e0f3fccbc6199a85dc

#utils
include("utils/tmaze.jl")
include("utils/learners.jl")

include("utils/experiment.jl")

end # module
