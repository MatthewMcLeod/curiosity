
using Random
using MinimalRLCore


"""
    Simple reward structure for goals. Can be duck typed.
"""
struct BasicRewFunc
    value::Float64
end

is_terminal(rew::BasicRewFunc) = true
get_reward(rew::BasicRewFunc) = rew.value


module ContGridWorldParams
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

ACTIONS = [UP, RIGHT, DOWN, LEFT]

const BODY_RADIUS = 0.2
const STEP = 0.5

const AGENT_BOUNDRIES = [
    [BODY_RADIUS, 0.0],
    [-BODY_RADIUS, 0.0],
    [0.0, BODY_RADIUS],
    [0.0, -BODY_RADIUS]
]
end

"""
    ContGridWorld
A Continuous grid world domain with generic dynamics to allow for different types of mazes.         
    - state: [y, x]
# Args
"""
mutable struct ContGridWorld{RF, SF} <: AbstractEnvironment
    # State is stored as (y,x)
    state::Array{Float64, 1}
    walls::Array{Bool, 2}

    # goals stored as (y,x)
    goals::Array{Int, 2}
    # reward_funcs::Dict{Int, F}
    cumulant_schedule::RF
    start_func::SF
    
    max_action_noise::Float64
    drift_noise::Float64
    normalized::Bool
    collision::Bool

    edge_locs::Array{Array{Int64, 1}, 1}
    collision_check::Array{Bool, 1}
    new_state::Array{Float64, 1}

    per_step_reward::Float64
    
    ContGridWorld(walls, goals,
                  cumulant_schedule::RF,# rew_funcs::Dict{Int, F},
                  start_func::SF, max_action_noise,
                  drift_noise, normalized, per_step_rew=0.0) where {RF, SF} =
                      new{RF, SF}([0.0, 0.0], walls, goals,
                                  cumulant_schedule,
                                  start_func,
                                  max_action_noise, drift_noise,
                                  normalized, false,
                                  [[0,0] for i in 1:4],
                                  fill(false, 4),
                                  zeros(2),
                                  per_step_rew)
end

ContGridWorld(walls, goals,
              # rew_funcs,
              max_action_noise,
              drift_noise, normalized, per_step_rew = 0.0) =
                  ContGridWorld(walls, goals,
                                nothing,
                                max_action_noise,
                                drift_noise,
                                normalized,
                                per_step_rew = 0.0)

MinimalRLCore.is_terminal(env::ContGridWorld) = begin
    goal_id = which_goal(env)
    if env.cumulant_schedule isa Dict
        if goal_id ∈ keys(env.cumulant_schedule)
            is_terminal(env.cumulant_schedule[goal_id])
        else
            false
        end
    elseif env.cumulant_schedule isa CumulantSchedule
        which_goal(env) > 0
    elseif env.cumulant_schedule isa nothing
        false
    end
end

MinimalRLCore.get_reward(env::ContGridWorld) = begin
    goal_id = which_goal(env)
    if env.cumulant_schedule isa Dict
        if goal_id ∈ keys(env.cumulant_schedule)
            get_reward(env.cumulant_schedule[goal_id])
        else
            env.per_step_reward
        end
    elseif env.cumulant_schedule isa CumulantSchedule
        env.per_step_reward
    elseif env.cumulant_schedule isa nothing
        env.per_step_reward
    end
end

function get_cumulants(env::ContGridWorld, cs::CumulantSchedule)
    goal_id = which_goal(env)
    cumulants = zeros(4)
    if goal_id == 1 #check_goal(env, 1)
        cumulants[1] = get_cumulant(cs, "G1")
    elseif goal_id == 2 #check_goal(env, 2)
        cumulants[2] = get_cumulant(cs, "G2")
    elseif goal_id == 3 #check_goal(env, 3)
        cumulants[3] = get_cumulant(cs, "G3")
    elseif goal_id == 4 #check_goal(env, 4)
        cumulants[4] = get_cumulant(cs, "G4")
    end
    cumulants
end

MinimalRLCore.get_state(env::ContGridWorld) = begin
    s = env.normalized ? env.state./size(env) : env.state
    if env.cumulant_schedule isa CumulantSchedule
        s = vcat(s, get_cumulants(env, env.cumulant_schedule))
    end
    s
end


random_state(env::ContGridWorld, rng) = [rand(rng)*size(env.walls)[1], rand(rng)*size(env.walls)[2]]
function random_start_state(env::ContGridWorld, rng)
    state = random_state(env, rng)
    while is_wall(env, state) || which_goal(env, state) > 0
        state = random_state(env, rng)
    end
    return state
end
Base.size(env::ContGridWorld) = size(env.walls)
num_actions(env::ContGridWorld) = 4
get_states(env::ContGridWorld) = findall(x->x==false, env.walls)
MinimalRLCore.get_actions(env::ContGridWorld) = ContGridWorldParams.ACTIONS


valid_projected_state(env::ContGridWorld, prj) = begin
    gw_size = size(env.walls)
    prj[1] >= 1 && prj[2] >= 1 && prj[1] <= gw_size[1]  && prj[2] <= gw_size[2]
end

project(env::ContGridWorld, state) = [Int64(floor(state[1])) + 1, Int64(floor(state[2])) + 1]
project(env::ContGridWorld, state, loc) = begin; loc[1] = Int64(floor(state[1]) + 1); loc[2] = Int64(floor(state[2]) + 1); end;

function is_wall(env::ContGridWorld, state::Array{Float64, 1})
    prj = project(env, state)
    return is_wall(env, prj)
end

function is_wall(env::ContGridWorld, prj::Array{Int64, 1})
    gw_size = size(env.walls)
    if !valid_projected_state(env, prj)
        return false
    end
    env.walls[prj[1], prj[2]]
end

which_goal(env::ContGridWorld) = which_goal(env, env.state)

function which_goal(env::ContGridWorld, state::Array{Float64, 1})
    prj = project(env, state)
    if !valid_projected_state(env, prj)
        return -1
    end
    env.goals[prj[1], prj[2]]
end

function handle_collision(env::ContGridWorld, state, action)

    #Approximate by the square...
    CRP = ContGridWorldParams
    new_state = copy(state)

    new_state[1] = clamp(new_state[1], CRP.BODY_RADIUS, size(env.walls)[1] - CRP.BODY_RADIUS)
    new_state[2] = clamp(new_state[2], CRP.BODY_RADIUS, size(env.walls)[2] - CRP.BODY_RADIUS)

    # Really basic collision detection for 2-d plane worlds.
    collided = new_state[1] != state[1] || new_state[2] != state[2]

    for i in 1:4
        project(env::ContGridWorld, new_state .+ CRP.AGENT_BOUNDRIES[i], env.edge_locs[i])
        env.collision_check[i] = is_wall(env, env.edge_locs[i])
    end

    collided = collided || any(env.collision_check)

    if env.collision_check[1] && env.collision_check[2]
        wall_piece = project(env, new_state)
        if action == CRP.DOWN
            new_state[1] = (wall_piece[1]) + CRP.BODY_RADIUS
        elseif action == CRP.UP
            new_state[1] = (wall_piece[1] - 1) - CRP.BODY_RADIUS
        end
    elseif env.collision_check[1]
        new_state[1] = (env.edge_locs[1][1] - 1) - CRP.BODY_RADIUS
    elseif env.collision_check[2]
        new_state[1] = (env.edge_locs[2][1]) + CRP.BODY_RADIUS
    end

    if env.collision_check[3] && env.collision_check[4]
        wall_piece = project(env, new_state)
        if action == CRP.LEFT
            new_state[2] = (wall_piece[2]) + CRP.BODY_RADIUS
        elseif action == CRP.RIGHT
            new_state[2] = (wall_piece[2] - 1) - CRP.BODY_RADIUS
        end
    elseif env.collision_check[3]
        new_state[2] = (env.edge_locs[3][2] - 1) - CRP.BODY_RADIUS
    elseif env.collision_check[4]
        new_state[2] = (env.edge_locs[4][2]) + CRP.BODY_RADIUS
    end

    new_state[1] = clamp(new_state[1], CRP.BODY_RADIUS, size(env.walls)[1] - CRP.BODY_RADIUS)
    new_state[2] = clamp(new_state[2], CRP.BODY_RADIUS, size(env.walls)[2] - CRP.BODY_RADIUS)

    return new_state, collided
end

function handle_collision!(env::ContGridWorld, action)
    env.state, collided = handle_collision(env, env.state, action)
    return collided
end

function MinimalRLCore.reset!(env::ContGridWorld, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    if env.start_func isa Nothing
        env.state = random_start_state(env, rng)
        env.state
    else
        env.state = env.start_func(env, rng)
        env.state
    end
end

function MinimalRLCore.reset!(env::ContGridWorld, state::AbstractArray)
    s = env.normalized ? state.*size(env) : state

    if is_wall(env, s)
        throw("Cannot reset environment to invalid state: $(state)")
    end

    env.state .= s
    # @show env.state
end

function mini_step!(env::ContGridWorld, step, action)
    env.state .+= step
    collision = handle_collision!(env, action)
    return collision
end

function MinimalRLCore.environment_step!(env::ContGridWorld, action, rng=Random.GLOBAL_RNG)

    CRP = ContGridWorldParams
    next_step = zeros(2)
    @assert action ∈ CRP.ACTIONS

    if action == CRP.UP
        next_step[1] = (CRP.STEP - rand(rng)*env.max_action_noise)
        next_step[2] = randn(rng)*env.drift_noise
    elseif action == CRP.DOWN
        next_step[1] = -(CRP.STEP - rand(rng)*env.max_action_noise)
        next_step[2] = randn(rng)*env.drift_noise
    elseif action == CRP.RIGHT
        next_step[1] = randn(rng)*env.drift_noise
        next_step[2] = CRP.STEP - rand(rng)*env.max_action_noise
    elseif action == CRP.LEFT
        next_step[1] = randn(rng)*env.drift_noise
        next_step[2] = -(CRP.STEP - rand(rng)*env.max_action_noise)
    end

    # mini_physics simulation for 1 second (== 10 steps of 0.1 seconds)
    Δt=1.0
    τ=10
    next_step .*= Δt/τ
    collision = false
    for t in 1:τ
        collision = mini_step!(env, next_step, action)
        if collision
            break
        end
    end
    env.collision = collision

    update!(env.cumulant_schedule, env.state)
end

module ContGridWorldPlotParams

using Colors
import ColorSchemes

const SIZE = 10
const BG = Colors.RGB(1.0, 1.0, 1.0)
const WALL = Colors.RGB(0.3, 0.3, 0.3)
const AC = Colors.RGB(0.69921875, 0.10546875, 0.10546875)
const GOAL = Colors.RGB(0.796875, 0.984375, 0.76953125)
const GOAL_PALETTE = ColorSchemes.tol_muted
const AGENT = [AC AC AC AC;
               AC AC AC AC;
               AC AC AC AC;
               AC AC AC AC;]

end

@recipe function f(env::ContGridWorld)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false

    PP = ContGridWorldPlotParams

    s = size(env)
    screen = fill(PP.BG, (s[2] + 2)*PP.SIZE, (s[1] + 2)*PP.SIZE)
    screen[:, 1:PP.SIZE] .= PP.WALL
    screen[1:PP.SIZE, :] .= PP.WALL
    screen[end-(PP.SIZE-1):end, :] .= PP.WALL
    screen[:, end-(PP.SIZE-1):end] .= PP.WALL
    
    for i ∈ 1:s[1]
        for j ∈ 1:s[2]
            sqr_i = ((i)*PP.SIZE + 1):((i+1)*PP.SIZE)
            sqr_j = ((j)*PP.SIZE + 1):((j+1)*PP.SIZE)
            if env.walls[s[1] - i + 1, j]
                screen[sqr_i, sqr_j] .= PP.WALL
            end
            if env.goals[s[1] - i + 1, j] > 0
                screen[sqr_i, sqr_j] .=
                    PP.GOAL_PALETTE[env.goals[s[1] - i + 1, j]]
            end
        end
    end
    state = ((env.state[1] - 1, env.state[2] + 1).*PP.SIZE)
    
    screen[collect(-1:2) .+ Int(floor(PP.SIZE*s[1] - state[1])), collect(-1:2) .+ Int(floor(state[2]))] .= PP.AGENT
    
    screen
end
