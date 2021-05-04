

include("cont_2d_worlds.jl")

module FourRoomsContParams

const BASE_WALLS = [0 0 0 0 0 1 0 0 0 0 0;
                    0 0 0 0 0 1 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 1 0 0 0 0 0;
                    0 0 0 0 0 1 0 0 0 0 0;
                    1 0 1 1 1 1 0 0 0 0 0;
                    0 0 0 0 0 1 1 1 0 1 1;
                    0 0 0 0 0 1 0 0 0 0 0;
                    0 0 0 0 0 1 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 1 0 0 0 0 0;]

const GOAL_LOCS =  [0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;
                    0 0 0 0 0 0 0 0 0 0 0;]

const ROOM_TOP_LEFT = 1
const ROOM_TOP_RIGHT = 2
const ROOM_BOTTOM_LEFT = 3
const ROOM_BOTTOM_RIGHT = 4

const REW_FUNCS = Dict{Int, Function}()

end


FourRooms(max_action_noise=0.1,
          drift_noise=0.001;
          normalized=false) =
              ContGridWorld(FourRoomsContParams.BASE_WALLS[end:-1:1, :],
                            FourRoomsContParams.GOAL_LOCS[end:-1:1, :],
                            FourRoomsContParams.REW_FUNCS,
                            max_action_noise,
                            drift_noise,
                            normalized)




module TMazeContParams

import ..BasicRewFunc

const BASE_WALLS = [1 1 1 1 1 1 1 1 1 1 1;
                    1 0 0 1 1 1 1 1 0 0 1;
                    1 0 0 0 0 0 0 0 0 0 1;
                    1 0 0 0 0 0 0 0 0 0 1;
                    1 0 0 0 0 0 0 0 0 0 1;
                    1 0 0 1 0 0 0 1 0 0 1;
                    1 1 1 1 0 0 0 1 1 1 1;
                    1 1 1 1 0 0 0 1 1 1 1;
                    1 1 1 1 0 0 0 1 1 1 1;
                    1 1 1 1 0 0 0 1 1 1 1;
                    1 1 1 1 0 0 0 1 1 1 1;]

const GOAL_LOCS = [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1;
                   -1  1  1 -1 -1 -1 -1 -1  3  3 -1;
                   -1  0  0  0  0  0  0  0  0  0 -1;
                   -1  0  0  0  0  0  0  0  0  0 -1;
                   -1  0  0  0  0  0  0  0  0  0 -1;
                   -1  2  2 -1  0  0  0 -1  4  4 -1;
                   -1 -1 -1 -1  0  0  0 -1 -1 -1 -1;
                   -1 -1 -1 -1  0  0  0 -1 -1 -1 -1;
                   -1 -1 -1 -1  0  0  0 -1 -1 -1 -1;
                   -1 -1 -1 -1  0  0  0 -1 -1 -1 -1;
                   -1 -1 -1 -1  0  0  0 -1 -1 -1 -1;]

const REW_FUNCS = Dict([i=>BasicRewFunc(i) for i in 1:4]...)

end

TMaze(max_action_noise=0.1, drift_noise=0.001; normalized=false) =
    ContGridWorld(TMazeContParams.BASE_WALLS[end:-1:1, :],
                  TMazeContParams.GOAL_LOCS[end:-1:1, :],
                  TMazeContParams.REW_FUNCS,
                  max_action_noise,
                  drift_noise,
                  normalized)


module OpenWorldContParams

import ..BasicRewFunc
import ..Distributions

const REW_FUNCS = Dict([i=>BasicRewFunc(i) for i in 1:4]...)

function center_start_func(env, rng)
    sze = size(env.walls)
    x_rng = Distributions.Uniform((round(sze[1]/2)) - 0.5, round(sze[1]/2) + 0.5)
    y_rng = Distributions.Uniform((round(sze[2]/2)) - 0.5, round(sze[2]/2) + 0.5)

    x = rand(rng, x_rng)
    y = rand(rng, y_rng)
    [x, y]
end

end

OpenWorld(width, height, max_action_noise=0.1, drift_noise=0.001; cumulant_schedule=nothing, normalized=true, start_type=:none) = begin
    base_walls = zeros(Int, width, height)
    goal_locs = zeros(Int, width, height)
    goal_locs[1, 1] = 1
    goal_locs[1, end] = 3
    goal_locs[end, 1] = 2
    goal_locs[end, end] = 4

    start = if start_type == :none
        nothing
    elseif start_type == :center
        OpenWorldContParams.center_start_func
    else
        throw("error")
    end
    
    ContGridWorld(base_walls,
                  goal_locs[end:-1:1, :],
                  cumulant_schedule,
                  start,
                  max_action_noise,
                  drift_noise,
                  normalized)
end
