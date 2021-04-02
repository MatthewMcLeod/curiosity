using Random
using RecipesBase

import MinimalRLCore
using Distributions: Uniform

# module OneDTmazeConst

# const FORWARD = 1
# const RIGHT = 2
# const BACKWARD = 3
# const LEFT = 4

# const ACTIONS = [FORWARD, RIGHT, BACKWARD, LEFT]

# end


"""
    1d TMaze

"""
Base.@kwdef struct OneDTMaze <: MinimalRLCore.AbstractEnvironment

    pos::Vector{Float64} = [0.0, 0.0]
    
    cumulant_schedule::CumulantSchedule = TMazeCumulantSchedules.Constant(1.0)
    starts::String = "beg"
    
    ACTION_STEP::Float64 = 0.1

    EPSILON::Float64 = 0.05
    
    UP::Int = 1
    RIGHT::Int = 2
    DOWN::Int = 3
    LEFT::Int = 4

    ACTIONS::Vector{Int} = [1,2,3,4]
end

OneDTMaze(starts::String) = OneDTMaze(starts=starts)
OneDTMaze(cs::CumulantSchedule, starts::String) = OneDTMaze(cumulant_schedule=cs, starts=starts)

volume(::OneDTMaze) = 0.8 + 1.0 + 2*0.4

MinimalRLCore.get_actions(env::OneDTMaze) = env.ACTIONS
valid_action(env::OneDTMaze, action) = action in MinimalRLCore.get_actions(env)
MinimalRLCore.get_reward(env::OneDTMaze) = 0.0 # -> determines if the agent_state is terminal
MinimalRLCore.is_terminal(env::OneDTMaze, pos=env.pos) = any(check_goal.([env], 1:4, [pos]))

function check_goal(env::OneDTMaze, goal, pos=env.pos)
    cur_x = pos[1]
    cur_y = pos[2]
    println("Goal Check $(env.pos), $(goal)")
    if goal == (goal isa String ? "G1" : 1)

        (cur_x == 0.0 && range_check(cur_y, 1.0-env.EPSILON, 1.0)) # G1
    elseif goal == (goal isa String ? "G2" : 2)
        (cur_x == 0.0 && range_check(cur_y, 0.0, 0.6+env.EPSILON)) # G2
    elseif goal == (goal isa String ? "G3" : 3)
        (cur_x == 1.0 && range_check(cur_y, 1.0-env.EPSILON, 1.0)) # G3
    elseif goal == (goal isa String ? "G4" : 4)
        (cur_x == 1.0 && range_check(cur_y, 0.0, 0.6+env.EPSILON)) # G4
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
        [0.5, rand(Uniform(0.0, 0.8))]
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
                [0.0, rand(Uniform(0.6, 1.0))]
            else
                [1.0, rand(Uniform(0.6, 1.0))]
            end
        end
    end
end

function MinimalRLCore.reset!(env::OneDTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    ns = get_random_state(env)
    env.pos .= ns
end

function MinimalRLCore.reset!(env::OneDTMaze,
                              start_state::Vector{Float64}) 
   env.pos .= start_state
end

function MinimalRLCore.environment_step!(env::OneDTMaze, action, rng::AbstractRNG=Random.GLOBAL_RNG)

    @boundscheck valid_action(env, action)
    x_mov_scale, y_mov_scale = if action == env.UP
        (0.0, env.ACTION_STEP)
    elseif action == env.DOWN
        (0.0, -env.ACTION_STEP)
    elseif action == env.RIGHT
        (env.ACTION_STEP, 0.0)
    elseif action == env.LEFT
        (-env.ACTION_STEP, 0.0)
    end

    x_mov, y_mov = rand(rng)*x_mov_scale, rand(rng)*y_mov_scale

    cur_x = env.pos[1]
    cur_y = env.pos[2]
    if (cur_x == 0.5 || cur_x == 0.0 || cur_x == 1.0) # in a vertical hallway
        if !range_check(cur_y, 0.8 - env.EPSILON, 0.8 + env.EPSILON) # can't go to horiz hallway
            x_mov = 0.0
        end
    elseif !range_check(cur_x, 0.0, 0.0 + env.EPSILON) ||
           !range_check(cur_x, 0.5 - env.EPSILON, 0.5 + env.EPSILON) || # In horizontal Hallway
           !range_check(cur_x, 1.0 - env.EPSILON, 1.0)
        y_mov = 0.0
    end

    env.pos[1] = clamp(x_mov + cur_x, 0.0, 1.0)
    if cur_x == 0.0 || cur_x == 1.0
        env.pos[2] = clamp(y_mov + cur_y, 0.6, 1.0)
    else
        env.pos[2] = clamp(y_mov + cur_y, 0.0, 0.8)
    end

end


