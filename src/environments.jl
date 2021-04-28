




abstract type CumulantSchedule end
function update! end
function get_cumulant(cs::CumulantSchedule, goal) end
function get_cumulant_eval_values end

#cumulants for TMaze
include("environments/tmaze_cumulants.jl")

# TMaze environments
include("environments/tabular_tmaze.jl")
include("environments/1d-tmaze.jl")

# Cont 2d Worlds environment.
include("environments/cont_2d_worlds_spec.jl")


# Mountain Car
include("environments/mountain_car.jl")


