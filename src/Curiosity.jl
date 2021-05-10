
module Curiosity

using Reexport
using GVFHordes

# abstract type Learner end

@reexport using MinimalRLCore

range_check(v, min, max) = v >= min && v <= max

using Flux
import Flux.Optimise: update!

# Abstract type of FeatureProjector used in the learner utils
abstract type FeatureCreator end
export ValueFeatureProjector, ActionValueFeatureProjector
include("utils/learners.jl")
include("utils/SRHorde.jl")
include("utils/dpi.jl")

export Auto
include("optimizers/Auto.jl")


export QLearner, LinearQLearner, VLearner, SRLearner, GPI, predict, predict_SF
include("learner.jl")

abstract type IntrinsicReward end
include("agent/intrinsic_rewards.jl")

export TabularRoundRobin, update!
include("updates/TabularRoundRobin.jl")

abstract type ExplorationStrategy end
export EpsilonGreedy, ϵGreedyDecay
include("agent/exploration.jl")

export Agent, agent_end!, step!, get_demon_prediction, μ_π, proc_input
include("agent/agent.jl")

export TileCoder, create_features
include("./agent/tile_coder.jl")


export TabularTMaze, MountainCar, OneDTMaze, valid_state_mask
include("environments.jl")


# logger
export Logger, logger_start!, logger_step!, logger_episode_end!, LoggerKey, LoggerInitKey
include("logger/logger.jl")

#utils
export get_active_action_state_vector, ValueFeatureProjector
include("utils/SRCreation.jl")
include("utils/tmaze.jl")
include("utils/1d-tmaze.jl")
include("utils/2d-gridworld.jl")
include("utils/mountain_car.jl")
include("utils/experiment.jl")
include("utils/agent.jl")
include("utils/features.jl")
include("utils/baseline_learners.jl")

using GVFHordes
export monte_carlo_returns
include("monte_carlo.jl")

end # module
