
module TabularTMazeCumulantSchedules

using Distributions
import ..TabularTMaze
import ..CumulantSchedule
import ..Curiosity: get_cumulants, update!

mutable struct DrifterDistractor <: CumulantSchedule
    constants::Float64
    drifter_std::Float64
    drifter_mean::Float64
    distractor_mean::Float64
    distractor_std::Float64
end

function get_cumulants(env::TabularTMaze, self::DrifterDistractor, pos)
    num_cumulants = 4
    cumulants = zeros(num_cumulants)
    if env.world[pos[1]][pos[2]] == "G1"
        cumulants[1] = rand(Normal(self.distractor_mean, self.distractor_std))
    elseif env.world[pos[1]][pos[2]] == "G2"
        cumulants[2] = self.constants
    elseif env.world[pos[1]][pos[2]] == "G3"
        cumulants[3] = self.drifter_mean
    elseif env.world[pos[1]][pos[2]] == "G4"
        cumulants[4] = self.constants
    end
    return cumulants
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

function update!(env::TabularTMaze, self::DrifterDistractor, pos)
    self.drifter_mean += rand(Normal(0, self.drifter_std))
end


mutable struct Constant <: CumulantSchedule
    c1::Float64
    c2::Float64
    c3::Float64
    c4::Float64
end

Constant(c = 1.0) = Constant(c,c,c,c)

function get_cumulants(env::TabularTMaze, cs::Constant, pos)
    num_cumulants = 4
    cumulants = zeros(num_cumulants)
    if env.world[pos[1]][pos[2]] == "G1"
        cumulants[1] = cs.c1
    elseif env.world[pos[1]][pos[2]] == "G2"
        cumulants[2] = cs.c2
    elseif env.world[pos[1]][pos[2]] == "G3"
        cumulants[3] = cs.c3
    elseif env.world[pos[1]][pos[2]] == "G4"
        cumulants[4] = cs.c4
    end
    return cumulants
end

function update!(env::TabularTMaze, cs::Constant, pos)
end

end
