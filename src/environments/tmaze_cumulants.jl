module TMazeCumulantConstants
    const MAX = 50
    const MIN = -50
end
module TMazeCumulantSchedules

using Distributions
import ..CumulantSchedule
import ..Curiosity: get_cumulant, update!, get_cumulant_eval_values
import ..TMazeCumulantConstants: MAX, MIN

mutable struct DrifterDistractor <: CumulantSchedule
    constant1::Float64
    constant2::Float64
    drifter_mean::Float64
    drifter_std::Float64
    distractor_mean::Float64
    distractor_std::Float64

    # function DrifterDistractor(constants, drifter_mean, drifter_std, distractor_mean, distractor_std)
    #     new(constants, constants, drifter_mean, drifter_std, distractor_mean, distractor_std)
    # end

end

function get_cumulant(cs::DrifterDistractor, goal::String)
    if goal == "G1"
        clamp(MIN, rand(Normal(cs.distractor_mean, cs.distractor_std)), MAX)
    elseif goal == "G2"
        cs.constant1
    elseif goal == "G3"
        cs.drifter_mean
    elseif goal == "G4"
        cs.constant2
    end
end

function get_cumulant_eval_values(self::DrifterDistractor)
    # Used for scaling the eval set based on the end values
    num_cumulants = 4
    cumulants = zeros(num_cumulants)
    cumulants[1] = self.distractor_mean
    cumulants[2] = self.constant1
    cumulants[3] = self.drifter_mean
    cumulants[4] = self.constant2
    return cumulants
end

# function update!(env::TabularTMaze, self::DrifterDistractor, pos)
function update!(self::DrifterDistractor, pos)
    self.drifter_mean += rand(Normal(0, self.drifter_std))
    self.drifter_mean = clamp(MIN,self.drifter_mean,MAX)
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

mutable struct DrifterDrifterDistractor <: CumulantSchedule
    constant1::Float64

    drifter_1_mean::Float64
    drifter_1_std::Float64
    drifter_2_mean::Float64
    drifter_2_std::Float64
    distractor_mean::Float64
    distractor_std::Float64

end

function get_cumulant(cs::DrifterDrifterDistractor, goal::String)
    if goal == "G1"
        clamp(MIN, rand(Normal(cs.distractor_mean, cs.distractor_std)), MAX)
    elseif goal == "G2"
        cs.constant1
    elseif goal == "G3"
        cs.drifter_1_mean
    elseif goal == "G4"
        cs.drifter_2_mean
    end
end

function get_cumulant_eval_values(self::DrifterDrifterDistractor)
    # Used for scaling the eval set based on the end values
    num_cumulants = 4
    cumulants = zeros(num_cumulants)
    cumulants[1] = self.distractor_mean
    cumulants[2] = self.constant1
    cumulants[3] = self.drifter_1_mean
    cumulants[4] = self.drifter_2_mean
    return cumulants
end

# function update!(env::TabularTMaze, self::DrifterDistractor, pos)
function update!(self::DrifterDrifterDistractor, pos)
    self.drifter_1_mean += rand(Normal(0, self.drifter_1_std))
    self.drifter_1_mean = clamp(MIN,self.drifter_1_mean,MAX)

    self.drifter_2_mean += rand(Normal(0, self.drifter_2_std))
    self.drifter_2_mean = clamp(MIN,self.drifter_2_mean,MAX)
end

end
