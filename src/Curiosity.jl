
module Curiosity

using Reexport
using GVFHordes

# abstract type Learner end

@reexport using MinimalRLCore

using Flux
import Flux.Optimise: update!

# Abstract type of FeatureProjector used in the learner utils
abstract type AbstractFeatureProjector end
export ValueFeatureProjector,ActionValueFeatureProjector
include("utils/learners.jl")
# export GVFHordes
# include("GVFHordes/GVFHordes.jl")
include("utils/SRHorde.jl")
include("utils/replay.jl")
include("utils/update_timer.jl")

export Auto
include("optimizers/Auto.jl")



export QLearner, LinearQLearner, VLearner, SRLearner, GPI, predict, predict_SF
include("learner.jl")

abstract type IntrinsicReward end
include("agent/intrinsic_rewards.jl")

export TabularRoundRobin, update!
include("updates/TabularRoundRobin.jl")

abstract type ExplorationStrategy end
export EpsilonGreedy
include("agent/exploration.jl")

export Agent, agent_end!, step!, get_demon_prediction
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
export get_active_action_state_vector, ValueFeatureProjector
include("utils/tmaze.jl")
include("utils/mountain_car.jl")
include("utils/experiment.jl")
include("utils/agent.jl")


using GVFHordes
export monte_carlo_returns
include("monte_carlo.jl")

end # module
