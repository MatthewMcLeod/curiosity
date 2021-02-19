
module TabularTMazeCumulantSchedules

using Distributions
import ..TabularTMaze
import ..CumulantSchedule
import ..Curiosity: get_cumulants, update!

# export get_cumulants, update!

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

function update!(env::TabularTMaze, self::DrifterDistractor, pos)
    self.drifter_mean += rand(Normal(0, self.drifter_std))
end

end
