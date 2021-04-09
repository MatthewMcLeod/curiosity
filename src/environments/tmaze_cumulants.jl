
module TMazeCumulantSchedules

using Distributions
import ..CumulantSchedule
import ..Curiosity: get_cumulant, update!, get_cumulant_eval_values

mutable struct DrifterDistractor <: CumulantSchedule
    constants::Float64
    drifter_std::Float64
    drifter_mean::Float64
    distractor_mean::Float64
    distractor_std::Float64
end

function get_cumulant(cs::DrifterDistractor, goal::String)
    if goal == "G1"
        rand(Normal(cs.distractor_mean, cs.distractor_std))
    elseif goal == "G2"
        cs.constants
    elseif goal == "G3"
        cs.drifter_mean
    elseif goal == "G4"
        cs.constants
    end
end

function get_cumulant_eval_values(self::DrifterDistractor)
    # Used for scaling the eval set based on the end values
    num_cumulants = 4
    cumulants = zeros(num_cumulants)
    cumulants[1] = self.distractor_mean
    cumulants[2] = self.constants
    cumulants[3] = self.drifter_mean
    cumulants[4] = self.constants
    return cumulants
end

# function update!(env::TabularTMaze, self::DrifterDistractor, pos)
function update!(self::DrifterDistractor, pos)
    self.drifter_mean += rand(Normal(0, self.drifter_std))
end


mutable struct Constant <: CumulantSchedule
    c1::Float64
    c2::Float64
    c3::Float64
    c4::Float64
end

Constant(c = 1.0) = Constant(c,c,c,c)

function get_cumulant(cs::Constant, goal::String)
    if goal == "G1"
        cs.c1
    elseif goal == "G2"
        cs.c2
    elseif goal == "G3"
        cs.c3
    elseif goal == "G4"
        cs.c4
    end
end

get_cumulant_eval_values(cs::Constant) = [cs.c1, cs.c2, cs.c3, cs.c4]

function update!(::Constant, pos)
end

end