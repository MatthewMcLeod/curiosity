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
    terminal_states::Array{Any,1}
    feature_size::Int64
    step::Int
    exploring_starts::Bool
    cumulant_schedule::CumulantSchedule

    function TabularTMaze(exploring_starts, cumulant_schedule)
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

        terminal_states = []
        start_state = [1,1]

        terms = [[1, 1], [5, 1], [1, 7], [5, 7]]
        for goal_state in terms
            obs = generate_obs(goal_state)
            push!(terminal_states, obs)
        end
        new(world, start_state, [1,1], goal_states, terminal_states, feature_size,step, exploring_starts, cumulant_schedule)
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

function is_terminal(env::TabularTMaze, pos)
    return env.world[pos[1]][pos[2]] in env.goal_states
end

function MinimalRLCore.reset!(environment::TabularTMaze, args...)
    if environment.exploring_starts == false
        environment.current_state = environment.start_state
    elseif environment.exploring_starts == true
        possible_start_states = findall(x -> x == "1", hcat(environment.world...))
        start_state = possible_start_states[rand(1:length(possible_start_states))]
        # The start states are flipped since hcat "transposes" a list of list when converting it to a matrix
        environment.current_state = [start_state[2],start_state[1]]
    end
end

function env_start!(environment::TabularTMaze)
    if environment.exploring_starts == false
        environment.current_state = environment.start_state
    elseif environment.exploring_starts == true
        possible_start_states = findall(x -> x == "1", hcat(environment.world...))
        start_state = possible_start_states[rand(1:length(possible_start_states))]
        # The start states are flipped since hcat "transposes" a list of list when converting it to a matrix
        environment.current_state = [start_state[2],start_state[1]]
    end
    obs = generate_obs(environment.current_state)
    cumulants = get_cumulants(environment, environment.cumulant_schedule, environment.current_state)

    return vcat(obs,cumulants), 0, false
end

function env_start!(environment::TabularTMaze, start_state)
    throw("Implement env_start with a start_state")
end

# function MinimalRLCore.environment_step!(env::TabularTMaze,
#                                    action,
#                                    rng; kwargs...)
#     _,_,_ = env_step!(env, action)
# end
function MinimalRLCore.environment_step!(env::TabularTMaze, action, rng::AbstractRNG=Random.GLOBAL_RNG)
    _,_,_ = env_step!(env, action)
end

function MinimalRLCore.get_reward(env::TabularTMaze)
    return 0.0
end


function MinimalRLCore.is_terminal(env::TabularTMaze) # -> determines if the agent_state is terminal
    return is_terminal(env, env.current_state)
end


function MinimalRLCore.get_state(env::TabularTMaze)
    obs = generate_obs(env.current_state)
    cumulants = get_cumulants(env, env.cumulant_schedule, env.current_state)

    return vcat(obs,cumulants)
end

function env_step!(environment::TabularTMaze, action)
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
    obs = generate_obs(environment.current_state)
    cumulants = get_cumulants(environment, environment.cumulant_schedule, environment.current_state)

    return vcat(obs,cumulants), reward, terminal
end

include("tabular_tmaze_cumulants.jl")
