using Random
using RecipesBase

import MinimalRLCore
using Distributions: Uniform

module OneDTmazeConst

const UP = 1
const RIGHT = 2
const DOWN = 3
const LEFT = 4

const ACTIONS = [UP, RIGHT, DOWN, LEFT]

const EPSILON = 0.0
const ACTION_STEP = 0.05

end


"""
    1d TMaze

"""
Base.@kwdef struct OneDTMaze <: MinimalRLCore.AbstractEnvironment

    pos::Vector{Float64} = [0.0, 0.0]
    
    cumulant_schedule::CumulantSchedule = TMazeCumulantSchedules.Constant(1.0)
    starts::String = "beg"

end

OneDTMaze(starts::String) = OneDTMaze(starts=starts)
OneDTMaze(cs::CumulantSchedule, starts::String) = OneDTMaze(cumulant_schedule=cs, starts=starts)

volume(::OneDTMaze) = 0.8 + 1.0 + 2*0.4

MinimalRLCore.get_actions(env::OneDTMaze) = OneDTmazeConst.ACTIONS
valid_action(env::OneDTMaze, action) = action in MinimalRLCore.get_actions(env)
MinimalRLCore.get_reward(env::OneDTMaze) = 0.0 # -> determines if the agent_state is terminal
MinimalRLCore.is_terminal(env::OneDTMaze, pos=env.pos) = any(check_goal.([env], 1:4, [pos]))

check_goal(env::OneDTMaze, goal, pos=env.pos) = check_goal(OneDTMaze, goal, pos)

function check_goal(env::Type{OneDTMaze}, goal, pos, epsilon=OneDTmazeConst.EPSILON)
    cur_x = pos[1]
    cur_y = pos[2]
    if goal == (goal isa String ? "G1" : 1)
        (cur_x == 0.0 && range_check(cur_y, 1.0-epsilon, 2.0)) # G1
    elseif goal == (goal isa String ? "G2" : 2)
        (cur_x == 0.0 && range_check(cur_y, 0.0, 0.6+epsilon)) # G2
    elseif goal == (goal isa String ? "G3" : 3)
        (cur_x == 1.0 && range_check(cur_y, 1.0-epsilon, 2.0)) # G3
    elseif goal == (goal isa String ? "G4" : 4)
        (cur_x == 1.0 && range_check(cur_y, 0.0, 0.6+epsilon)) # G4
    end
end

function get_cumulants(env::OneDTMaze, cs::CumulantSchedule)
    cumulants = zeros(4)
    if check_goal(env, 1)
        cumulants[1] = get_cumulant(cs, "G1")
    elseif check_goal(env, 2)
        cumulants[2] = get_cumulant(cs, "G2")
    elseif check_goal(env, 3)
        cumulants[3] = get_cumulant(cs, "G3")
    elseif check_goal(env, 4)
        cumulants[4] = get_cumulant(cs, "G4")
    end
    cumulants
end

function MinimalRLCore.get_state(env::OneDTMaze)
    cumulants = get_cumulants(env, env.cumulant_schedule)
    return vcat(env.pos, cumulants)
end

function get_random_state(env::OneDTMaze)

    if env.starts == "beg"
        [0.5, rand(Uniform(0.0, 0.1))]
    elseif env.starts == "const"
        [0.5, 0.0]
    elseif env.starts == "whole"
        h = rand()
        if h < 0.8/volume(env)
            [0.5, rand(Uniform(0.0, 0.8))]
        elseif h < 1.8/volume(env)
            [rand(Uniform(0.0, 1.0)), 0.8]
        else
            if rand() > 0.5
                [0.0, rand(Uniform(0.65, 0.95))]
            else
                [1.0, rand(Uniform(0.65, 0.95))]
            end
        end
    end
end

function MinimalRLCore.reset!(env::OneDTMaze)
    ns = get_random_state(env)
    env.pos .= ns
end

function MinimalRLCore.reset!(env::OneDTMaze,
                              start_state::Vector{Float64}) 
   env.pos .= start_state
end

function MinimalRLCore.environment_step!(env::OneDTMaze, action, rng::AbstractRNG=Random.GLOBAL_RNG)
    ODTMC = OneDTmazeConst
    @boundscheck valid_action(env, action)
    rand_mov = rand(rng, Uniform(-0.01, 0.01))
    x_mov, y_mov = if action == ODTMC.UP
        (0.0, ODTMC.ACTION_STEP + rand_mov)
    elseif action == ODTMC.DOWN
        (0.0, -ODTMC.ACTION_STEP + rand_mov)
    elseif action == ODTMC.RIGHT
        (ODTMC.ACTION_STEP + rand_mov, 0.0)
    elseif action == ODTMC.LEFT
        (-ODTMC.ACTION_STEP + rand_mov, 0.0)
    end

    cur_x = env.pos[1]
    cur_y = env.pos[2]
    if (cur_x == 0.5 || cur_x == 0.0 || cur_x == 1.0) # in a vertical hallway
        if !range_check(cur_y, 0.8 - ODTMC.EPSILON, 0.8 + ODTMC.EPSILON) # can't go to horiz hallway
            x_mov = 0.0
        elseif ODTMC.LEFT == action || ODTMC.RIGHT == action
            cur_y = 0.8
        end
    elseif !(range_check(cur_x, 0.0, 0.0 + ODTMC.EPSILON) ||
             range_check(cur_x, 0.5 - ODTMC.EPSILON, 0.5 + ODTMC.EPSILON) || # In horizontal Hallway
             range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0))
        y_mov = 0.0
    elseif range_check(cur_x, 0.0, 0.0 + ODTMC.EPSILON)
        cur_x = 0.0
    elseif range_check(cur_x, 1.0 - ODTMC.EPSILON, 1.0)
        if ODTMC.UP == action || ODTMC.DOWN == action
            cur_x = 1.0
        end
    end

    env.pos[1] = clamp(x_mov + cur_x, 0.0, 1.0)
    if env.pos[1] == 0.0 || env.pos[1] == 1.0
        env.pos[2] = clamp(y_mov + cur_y, 0.6, 1.0)
    else
        env.pos[2] = clamp(y_mov + cur_y, 0.0, 0.8)
    end

    
    update!(env.cumulant_schedule, env.pos)

end

module OneDTMazePlotParams

using Colors
import ColorSchemes

const SIZE = 10
const BG = Colors.RGB(1.0, 1.0, 1.0)
const WALL = Colors.RGB(0.3, 0.3, 0.3)
const AC = Colors.RGB(0.69921875, 0.10546875, 0.10546875)
const GOAL = Colors.RGB(0.796875, 0.984375, 0.76953125)
const GOAL_PALETTE = ColorSchemes.tol_muted
const AGENT = fill(AC, SIZE, SIZE)

end

@recipe function f(env::OneDTMaze)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false

    PP = OneDTMazePlotParams

    screen = fill(PP.WALL, PP.SIZE*12, PP.SIZE*12)

    # Paths
    screen[Int(PP.SIZE-PP.SIZE//2+1):Int(9*PP.SIZE + PP.SIZE//2), (Int(11//2*PP.SIZE) + 1):(Int((13//2)*PP.SIZE))] .= PP.BG
    screen[(Int(17//2*PP.SIZE) + 1):Int(19//2*PP.SIZE), Int(PP.SIZE-PP.SIZE//2+1):Int(11*PP.SIZE + PP.SIZE//2)] .= PP.BG

    screen[Int(14//2*PP.SIZE + 1):Int(22//2*PP.SIZE), Int(1//2*PP.SIZE+1):Int(3//2*PP.SIZE)] .= PP.BG
    screen[Int(14//2*PP.SIZE + 1):Int(22//2*PP.SIZE), Int(21//2*PP.SIZE+1):Int(23//2*PP.SIZE)] .= PP.BG

    # GOALS
    ϵ = OneDTmazeConst.EPSILON
    if OneDTmazeConst.EPSILON == 0.0
        screen[Int(13//2*PP.SIZE+1):Int(14//2*PP.SIZE), Int(1//2*PP.SIZE+1):Int(3//2*PP.SIZE)] .= PP.GOAL_PALETTE[2]
        screen[Int(22//2*PP.SIZE+1):Int(23//2*PP.SIZE), Int(1//2*PP.SIZE+1):Int(3//2*PP.SIZE)] .= PP.GOAL_PALETTE[1]
        screen[Int(13//2*PP.SIZE+1):Int(14//2*PP.SIZE), Int(21//2*PP.SIZE+1):Int(23//2*PP.SIZE)] .= PP.GOAL_PALETTE[4]
        screen[Int(22//2*PP.SIZE+1):Int(23//2*PP.SIZE), Int(21//2*PP.SIZE+1):Int(23//2*PP.SIZE)] .= PP.GOAL_PALETTE[3]
    else
        screen[Int(13//2*PP.SIZE+1):Int(14//2*PP.SIZE+PP.SIZE*ϵ), Int(1//2*PP.SIZE+1):Int(3//2*PP.SIZE)] .= PP.GOAL_PALETTE[2]
        screen[Int(22//2*PP.SIZE-PP.SIZE*ϵ+1):Int(23//2*PP.SIZE), Int(1//2*PP.SIZE+1):Int(3//2*PP.SIZE)] .= PP.GOAL_PALETTE[1]
        screen[Int(13//2*PP.SIZE+1):Int(14//2*PP.SIZE+PP.SIZE*ϵ), Int(21//2*PP.SIZE+1):Int(23//2*PP.SIZE)] .= PP.GOAL_PALETTE[4]
        screen[Int(22//2*PP.SIZE-PP.SIZE*ϵ+1):Int(23//2*PP.SIZE), Int(21//2*PP.SIZE+1):Int(23//2*PP.SIZE)] .= PP.GOAL_PALETTE[3]
    end
    
    state = env.pos
    x, y = Int(floor((env.pos[1]*10 + 1)*PP.SIZE)), Int(floor((env.pos[2]*10 + 1)*PP.SIZE))

    screen[Int(y-PP.SIZE//5 + 1):Int(y+PP.SIZE//5), Int(x-PP.SIZE//2 + 1):Int(x+PP.SIZE//2)] .= PP.AC
    screen[Int(y-PP.SIZE//2 + 1):Int(y+PP.SIZE//2), Int(x-PP.SIZE//5 + 1):Int(x+PP.SIZE//5)] .= PP.AC


    
    screen[end:-1:1, :]
end
