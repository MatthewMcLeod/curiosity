module Curiosity

abstract type Learner end

export TB, TBAuto, update!
include("learners/TB.jl")
include("learners/TB_Auto.jl")


export TabularRoundRobin, update!
include("learners/TabularRoundRobin.jl")


export Agent, agent_end!, step!
include("agent/agent.jl")


abstract type CumulantSchedule end
function update! end
function get_cumulants end

export TabularTMaze, env_step!, env_start!, valid_state_mask
    # TabularMazeCumulantSchedules, get_cumulants, update!
include("environments/tabular_tmaze.jl")

#utils
include("utils/tmaze.jl")
include("utils/learners.jl")

end # module
