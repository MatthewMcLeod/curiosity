using Random
using Distributions
using Statistics
using SparseArrays
using MinimalRLCore


mutable struct TabularTMaze <: MinimalRLCore.AbstractEnvironment
    world::Array{Array{String}}
    start_state::Array{Int64}
    current_state::Array{Int64}
    goal_states::Array{String}
    feature_size::Int64
    step::Int
    exploring_starts::Bool
    cumulant_schedule::CumulantSchedule
    extrinsic_reward::Float64

    function TabularTMaze(exploring_starts, cumulant_schedule; extrinsic_reward = -0.01)
        feature_size = 21
        num_actions = 4
        step = 1

        world = [["G1", "0", "0", "0", "0", "0", "G3"],
                 ["1", "0", "0", "0", "0", "0", "1"],
                 ["1", "1", "1", "1", "1", "1", "1"],
                 ["1", "0", "0", "1", "0", "0", "1"],
                 ["G2", "0", "0", "1", "0", "0", "G4"],
                 ["0", "0", "0", "1", "0", "0", "0"],
                 ["0", "0", "0", "1", "0", "0", "0"],
                 ["0", "0", "0", "1", "0", "0", "0"],
                 ["0", "0", "0", "1", "0", "0", "0"]]
        goal_states = ["G1", "G2", "G3", "G4"]
        start_state = [9,4]
        new(world, start_state, [9,4], goal_states, feature_size,step, exploring_starts, cumulant_schedule, extrinsic_reward)
    end
end

Base.size(e::TabularTMaze) = e.feature_size


function generate_obs(state::Array{Int64})
    obs = zeros(Int64, 9, 7)
    obs[state[1],state[2]] = 1
    toReturn = findfirst(!iszero, obs[valid_state_mask()])
    return toReturn
end

function valid_state_mask()
    """
    return a mask of valid states that is 9x7
    """
    world = [["G1", "0", "0", "0", "0", "0", "G3"],
             ["1", "0", "0", "0", "0", "0", "1"],
             ["1", "1", "1", "1", "1", "1", "1"],
             ["1", "0", "0", "1", "0", "0", "1"],
             ["G2", "0", "0", "1", "0", "0", "G4"],
             ["0", "0", "0", "1", "0", "0", "0"],
             ["0", "0", "0", "1", "0", "0", "0"],
             ["0", "0", "0", "1", "0", "0", "0"],
             ["0", "0", "0", "1", "0", "0", "0"]]

    world = permutedims(hcat(world...))
    valid_states = findall(x-> x!="0", world)
    return valid_states
end

function valid_state_action_mask()
"""
returns a mask for the state-action mask (63,4)
"""
    num_actions = 4
    world = [["G1", "0", "0", "0", "0", "0", "G3"],
                      ["1", "0", "0", "0", "0", "0", "1"],
                      ["1", "1", "1", "1", "1", "1", "1"],
                      ["1", "0", "0", "1", "0", "0", "1"],
                      ["G2", "0", "0", "1", "0", "0", "G4"],
                      ["0", "0", "0", "1", "0", "0", "0"],
                      ["0", "0", "0", "1", "0", "0", "0"],
                      ["0", "0", "0", "1", "0", "0", "0"],
                      ["0", "0", "0", "1", "0", "0", "0"]]

    flattened_world = reshape(permutedims(hcat(world...)),(63))
    valid_states = findall(x-> x!="0", flattened_world)
    return valid_states
end


MinimalRLCore.get_reward(env::TabularTMaze) = env.extrinsic_reward
is_terminal(env::TabularTMaze, pos) = env.world[pos[1]][pos[2]] in env.goal_states
MinimalRLCore.is_terminal(env::TabularTMaze) = is_terminal(env, env.current_state)
MinimalRLCore.get_actions(env::TabularTMaze) = 1:4

function MinimalRLCore.get_state(env::TabularTMaze)
    obs = generate_obs(env.current_state)
    cumulants = get_cumulants(env, env.cumulant_schedule, env.current_state)

    return vcat(obs,cumulants)
end

function get_cumulants(env::TabularTMaze, cs::CumulantSchedule, pos)
    num_cumulants = 4
    cumulants = zeros(num_cumulants)
    if env.world[pos[1]][pos[2]] == "G1"
        cumulants[1] = get_cumulant(cs, "G1")
    elseif env.world[pos[1]][pos[2]] == "G2"
        cumulants[2] = get_cumulant(cs, "G2")
    elseif env.world[pos[1]][pos[2]] == "G3"
        cumulants[3] = get_cumulant(cs, "G3")
    elseif env.world[pos[1]][pos[2]] == "G4"
        cumulants[4] = get_cumulant(cs, "G4")
    end
    return cumulants
end

function MinimalRLCore.reset!(environment::TabularTMaze, rng::AbstractRNG=Random.GLOBAL_RNG)
    if environment.exploring_starts == false
        environment.current_state = environment.start_state
    elseif environment.exploring_starts == true
        possible_start_states = findall(x -> x == "1", hcat(environment.world...))
        start_state = possible_start_states[rand(rng, 1:length(possible_start_states))]
        # The start states are flipped since hcat "transposes" a list of list when converting it to a matrix
        environment.current_state = [start_state[2],start_state[1]]
    end
end

function MinimalRLCore.reset!(environment::TabularTMaze, start_state::CartesianIndex)
    # throw("Implement env_start with a start_state")
    # environment.current_state = [start_state[2], start_state[1]]
    environment.current_state = [start_state[1], start_state[2]]
end

function MinimalRLCore.environment_step!(environment::TabularTMaze, action, rng::AbstractRNG=Random.GLOBAL_RNG)
    actions = [(-1, 0), (0, 1), (1, 0), (0, -1)] # up, right, down, left
    reward = 0.0
    terminal = false

    potential_state = [environment.current_state[1] + actions[action][1],
                       environment.current_state[2] + actions[action][2]]

    if potential_state[1] in [0, 10] || potential_state[2] in [0, 8] # No nothing, agent doesn't move
    elseif environment.world[potential_state[1]][potential_state[2]] == "1"
        environment.current_state = potential_state
    elseif is_terminal(environment, potential_state)
        environment.current_state = potential_state
        terminal = true
    end
    # update!(environment, environment.cumulant_schedule, environment.current_state)
    update!(environment.cumulant_schedule, environment.current_state)
end

using RecipesBase, Colors
@recipe function f(env::TabularTMaze)
    ticks := nothing
    foreground_color_border := nothing
    grid := false
    legend := false
    aspect_ratio := 1
    xaxis := false
    yaxis := false
    yflip := false

    SIZE=20
    BG = Colors.RGB(1.0, 1.0, 1.0)
    BORDER = Colors.RGB(0.0, 0.0, 0.0)
    WALL = Colors.RGB(0.3, 0.3, 0.3)
    AC = Colors.RGB(0.69921875, 0.10546875, 0.10546875)
    GOAL = Colors.RGB(0.796875, 0.984375, 0.76953125)

    cell = fill(BG, SIZE, SIZE)
    cell[1, :] .= BORDER
    cell[end, :] .= BORDER
    cell[:, 1] .= BORDER
    cell[:, end] .= BORDER


    s_y = length(env.world)
    s_x = length(env.world[1])

    screen = fill(BG, (s_y + 2)*SIZE, (s_x + 2)*SIZE)

    screen[:, 1:SIZE] .= WALL
    screen[1:SIZE, :] .= WALL
    screen[end-(SIZE-1):end, :] .= WALL
    screen[:, end-(SIZE-1):end] .= WALL

    for j ∈ 1:s_x
        for i ∈ 1:s_y
            sqr_i = ((i)*SIZE + 1):((i+1)*SIZE)
            sqr_j = ((j)*SIZE + 1):((j+1)*SIZE)
            if env.current_state[1] == i && env.current_state[2] == j
                v = @view screen[sqr_i, sqr_j]
                v .= cell
                v[Int(SIZE/2)-4:Int(SIZE/2)+5, Int(SIZE/2)-4:Int(SIZE/2)+5] .= AC
            elseif env.world[i][j] == "0"
                screen[sqr_i, sqr_j] .= WALL
            elseif env.world[i][j] == "1"
                screen[sqr_i, sqr_j] .= cell
            elseif env.world[i][j][1] == 'G'
                v = @view screen[sqr_i, sqr_j]
                v .= GOAL
                v[1, :] .= BORDER
                v[:, 1] .= BORDER
                v[end, :] .= BORDER
                v[:, end] .= BORDER
            end
        end
    end
    screen[end:-1:1,:]
end
