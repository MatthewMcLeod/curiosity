### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ a7871567-172a-42ec-b889-56c5440c99fa
begin
	using Revise
	using ProgressMeter
end

# ╔═╡ 7fa178d6-9f1c-11eb-3604-9777d8da228e
using Curiosity, Plots, PlutoUI, Statistics, Random

# ╔═╡ 60fa4e34-de7a-43c3-91d6-e4f60f1ca272
include("../experiments/tabular_tmaze.jl")

# ╔═╡ 3d63f4ea-a20a-438d-a359-35a7f84b789e
TTMU = Curiosity.TabularTMazeUtils

# ╔═╡ 6f96b22d-1bbe-4711-b4bd-b9b89130fead
default_args(ϵ) =
    Dict(
        # Behaviour Items
        "behaviour_eta" => 0.04,
        "behaviour_gamma" => 0.9,
        "behaviour_learner" => "GPI",
        "behaviour_update" => "TB",
        "behaviour_trace" => "AccumulatingTraces",
        "behaviour_opt" => "Auto",
        "behaviour_lambda" => 0.9,
        "behaviour_alpha_init" => 0.1,
        "exploration_param" => ϵ,
        "exploration_strategy" => "epsilon_greedy",

        # Demon Attributes
        "demon_alpha_init" => 0.1,
        "demon_eta" => 0.5,
        "demon_discounts" => 0.9,
        "demon_learner" => "SR",
        "demon_update" => "TB",
        "demon_policy_type" => "greedy_to_cumulant",
        "demon_opt" => "Auto",
        "demon_lambda" => 0.9,
        "demon_trace"=> "AccumulatingTraces",
        "demon_beta_m" => 0.9,
        "demon_beta_v" => 0.99,

        # Environment Config
        "constant_target"=> 1.0,
        "cumulant_schedule" => "DrifterDistractor",
        "distractor" => (1.0, 5.0),
        "drifter" => (1.0, sqrt(0.01)),
        "exploring_starts"=>false,

        # Agent and Logger
        "horde_type" => "regular",
        "intrinsic_reward" => "weight_change",
        "logger_keys" => [LoggerKey.TTMAZE_ERROR, LoggerKey.TTMAZE_UNIFORM_ERROR, LoggerKey.TTMAZE_OLD_ERROR],
        "save_dir" => "TabularTMazeExperiment",
        "seed" => 1,
        "steps" => 10000,
        "use_external_reward" => true,
        "logger_interval" => 100,
    )

# ╔═╡ b1573f0a-6622-4a40-b171-24f8afae3505
env = TabularTMaze(true, Curiosity.TMazeCumulantSchedules.Constant(1))

# ╔═╡ c655ab97-c20c-4e62-8435-533d50f47ade
begin
	start!(env)
	plot(env)
end

# ╔═╡ 037b8feb-c1de-4c5e-a8f4-fbd6b762673e
function get_policy_arrows(env,agent)
	#HARDCODED CAUSE IDK WHAT ELSE TO DO :(
	s_x = 7
	s_y = 9
	SIZE = 20
	OFFSET = 1*SIZE
	arrow_x = []
	arrow_y = []
	arrow_u = []
	arrow_v = []
	function generate_obs(state::Array{Int64})
		obs = zeros(Int64, 9, 7)
		obs[state[1],state[2]] = 1
		toReturn = findfirst(!iszero, obs[Curiosity.valid_state_mask()])
		return toReturn
	end
	for j ∈ 1:s_x
		for i ∈ 1:s_y
			sqr_i = ((i)*SIZE + 1):((i+1)*SIZE)
			sqr_j = ((j)*SIZE + 1):((j+1)*SIZE)

			k = generate_obs([i,j])
			if k != nothing # If it is not part of the 21 valid cells
				# @show k,i,j
				obs = zeros(5)
				obs[1] = k
				b_π = Curiosity.μ_π(agent, obs)
				# val, ACTION = findmax(b_π)

				# drawarray is bugged. Need to have it off. This is the correct transformation
				# when yflip=true
				# https://github.com/JuliaPlots/Plots.jl/issues/1498
				x_coord = OFFSET + (j-1) * SIZE + OFFSET/2
				y_coord = s_y * SIZE + 2*OFFSET - i * SIZE - OFFSET/2
				# yflip = false
				# x_coord = s_x * SIZE + 2 * OFFSET - (j) * SIZE - OFFSET/2
				# y_coord = OFFSET + (i-1) * SIZE + OFFSET/2

				max_prob,_ = findmax(b_π)
				for (ACTION,π) in enumerate(b_π)
					if π != max_prob
						continue
					end

					if ACTION == 1
						u,v = 0,1
					elseif ACTION == 2
						u,v, = 1,0
					elseif ACTION == 3
						u,v = 0,-1
					elseif ACTION == 4
						u,v = -1,0
					end
					push!(arrow_x,x_coord)
					push!(arrow_y,y_coord)
					push!(arrow_u, u*SIZE/2)
					push!(arrow_v,v*SIZE/2)
				end
			end
		end
	end
	return arrow_x,arrow_y,arrow_u,arrow_v
end

# ╔═╡ 77dae42d-3be5-43b0-93b0-611fa8c9f123
function main_experiment(parsed=default_args(0.1); progress=false, working=false, show_policy = false)

    num_steps = parsed["steps"]
    Random.seed!(parsed["seed"])

    cumulant_schedule = TTMU.get_cumulant_schedule(parsed)
    exploring_starts = parsed["exploring_starts"]
    env = TabularTMaze(exploring_starts, cumulant_schedule)

    agent = TabularTMazeExperiment.construct_agent(parsed)

    eps = 1
    max_num_steps = num_steps
    steps = Int[]

    if progress
        p = Progress(max_num_steps)
    end
	anim = Animation()
    while sum(steps) < max_num_steps
        cur_step = 0
        max_episode_steps = max_num_steps - sum(steps)
        tr, stp =
            run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
                #This is a callback for every timestep where logger can go
                # agent is accesible in this scope
				plot(env, title=string("Eps: ", parsed["exploration_param"],
					" Step: ",cur_step,
					" Ep: ", eps,
					" DL: ", parsed["demon_learner"],
					" BL: ", parsed["behaviour_learner"]))
				if show_policy
					arrow_x, arrow_y, arrow_u, arrow_v = get_policy_arrows(env,agent)
					quiver!(arrow_x,arrow_y,quiver=(arrow_u,arrow_v))
				end
				frame(anim)
                cur_step+=1
            end
        push!(steps, stp)
        eps += 1

        if progress
            ProgressMeter.update!(p, sum(steps))
        end
    end
	anim
end


# ╔═╡ 465bd0cd-3204-41a2-a68c-f3e38cdd0b5f
anim = main_experiment(default_args(0.2), show_policy=true, progress=true)

# ╔═╡ 1bacedb7-8373-4d52-99b8-54e1e5460346
# let
# 	for ϵ ∈ [0.1, 0.2, 0.3, 0.4]
# 		anim = main_experiment(default_args(ϵ))
# 		mp4(anim, "GPI_SR_epsilon_$(ϵ).mp4")
# 	end
# end

# ╔═╡ Cell order:
# ╠═a7871567-172a-42ec-b889-56c5440c99fa
# ╠═7fa178d6-9f1c-11eb-3604-9777d8da228e
# ╠═3d63f4ea-a20a-438d-a359-35a7f84b789e
# ╠═60fa4e34-de7a-43c3-91d6-e4f60f1ca272
# ╠═6f96b22d-1bbe-4711-b4bd-b9b89130fead
# ╟─b1573f0a-6622-4a40-b171-24f8afae3505
# ╠═c655ab97-c20c-4e62-8435-533d50f47ade
# ╠═037b8feb-c1de-4c5e-a8f4-fbd6b762673e
# ╟─77dae42d-3be5-43b0-93b0-611fa8c9f123
# ╠═465bd0cd-3204-41a2-a68c-f3e38cdd0b5f
# ╠═1bacedb7-8373-4d52-99b8-54e1e5460346
