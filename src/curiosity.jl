module Curiosity

abstract type Learner end
include("learners/TB.jl")
export TB, update!

include("learners/TabularRoundRobin.jl")
export TabularRoundRobin, update!

include("agent/agent.jl")
export Agent, agent_end!, step!

abstract type CumulantSchedule end 
include("environments/tabular_tmaze.jl")
export TabularTMaze, env_step!, env_start!

# abstract type CumulantSchedule end
include("environments/tabular_tmaze_drifter_distractor.jl")
export TabularTMazeDrifterDistractor, get_cumulants, update!


end # module
