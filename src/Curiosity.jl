
module Curiosity

using Reexport
using GVFHordes

abstract type Learner end

@reexport using MinimalRLCore


export Auto
include("optimizers/Auto.jl")

export TB, TBAuto, ESARSA, SR, update!, predict, predict_SF
include("learners/TB.jl")
include("learners/TB_Auto.jl")
include("learners/ESARSA.jl")
include("learners/SR.jl")

abstract type IntrinsicReward end
include("agent/intrinsic_rewards.jl")

export TabularRoundRobin, update!
include("learners/TabularRoundRobin.jl")

export Agent, agent_end!, step!
include("agent/agent.jl")

export TileCoder, create_features
include("./agent/tile_coder.jl")

abstract type CumulantSchedule end
function update! end
function get_cumulants end

export TabularTMaze, MountainCar, valid_state_mask
include("environments/tabular_tmaze.jl")
include("environments/mountain_car.jl")

# logger
export Logger, logger_step!, logger_episode_end!, LoggerKey, LoggerInitKey
include("logger/logger.jl")

#utils
include("utils/SRHorde.jl")
include("utils/tmaze.jl")
include("utils/mountain_car.jl")
include("utils/learners.jl")
include("utils/experiment.jl")


using GVFHordes
export monte_carlo_returns
include("monte_carlo.jl")

end # module
