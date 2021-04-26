### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# ╔═╡ cf8d2544-a693-11eb-298c-1109f75c63d2
using Curiosity, Plots, PlutoUI, Statistics, Random

# ╔═╡ 21c4a049-a940-4a37-8b6b-a3e09588e98c
include("./experiments/1d-tmaze.jl")

# ╔═╡ a32cdc08-b633-41d2-9e2a-555ea9442c49
ODTMU = Curiosity.OneDTMazeUtils

# ╔═╡ dc8ae3fb-7aad-4449-a7de-55b22a9bb110

default_args(ϵ) =
Dict(
	"logger_interval" => 100,

	# Behaviour Items
	# "behaviour_eta" => 0.1/8,
	"behaviour_gamma" => 0.9,
	"behaviour_learner" => "RoundRobin",
	"behaviour_update" => "ESARSA",
	"behaviour_reward_projector" => "base",
	"behaviour_rp_tilings" => 1,
	"behaviour_rp_tiles" => 16,
	"behaviour_trace" => "AccumulatingTraces",
	"behaviour_opt" => "Descent",
	"behaviour_lambda" => 0.9,
	"exploration_param" => ϵ,
	"exploration_strategy" => "epsilon_greedy",
	"ϵ_range" => (0.4,0.1),
	"decay_period" => 5000,
	"warmup_steps" => 1000,
	"behaviour_w_init" => 10,

	# Demon Attributes
	"demon_alpha_init" => 0.1,
	# "demon_eta" => 0.1/8,
	"demon_discounts" => 0.9,
	"demon_learner" => "Q",
	"demon_update" => "TB",
	"demon_policy_type" => "greedy_to_cumulant",
	"demon_opt" => "Auto",
	"demon_lambda" => 0.9,
	"demon_trace"=> "AccumulatingTraces",
	"demon_beta_m" => 0.99,
	"demon_beta_v" => 0.99,

	#shared
	"num_tiles" => 2,
	"num_tilings" =>8,
	"demon_rep" => "ideal_martha",
	# "demon_rep" => "tilecoding",
	"demon_num_tiles" => 6,
	"demon_num_tilings" => 1,
	"eta" => 0.05,

	# Environment Config
	"constant_target"=> [-10.0,10.0],
	"cumulant_schedule" => "DrifterDistractor",
	"distractor" => (1.0, 5.0),
	"drifter" => (1.0, sqrt(0.01)),
	"exploring_starts"=>"beg",
	"env_step_penalty" => -0.005,


	# Agent and Logger
	"horde_type" => "regular",
	"intrinsic_reward" => "weight_change",
	# "logger_keys" => [LoggerKey.TTMAZE_ERROR],
	"save_dir" => "OneDTMazeExperiment",
	"seed" => 1,
	"steps" => 100,
	"use_external_reward" => true,

	"logger_keys" => [LoggerKey.ONEDTMAZEERROR, LoggerKey.ONED_GOAL_VISITATION, LoggerKey.EPISODE_LENGTH, LoggerKey.INTRINSIC_REWARD]
)

# ╔═╡ fce35f63-e4b0-4d4f-b287-936ad5dc4f5e
env = OneDTMaze(Curiosity.TMazeCumulantSchedules.Constant(1), "beg",-0.05)

# ╔═╡ ce948feb-dfff-4337-8f1f-ce72b4028e54
begin 
	@userplot OneDTMaze
	@recipe function f(env::OneDTMaze)
		p = scatter(env.pos[1:1],env.pos[2:2])
		plot!([0.5,0.5],[0.0,0.8], label="", color=:black)
		plot!([0.0,0.0],[0.6,1.0], label="", color=:black)
		plot!([1.0,1.0],[0.6,1.0], label="", color=:black)
		plot!([0.0,1.0],[0.8,0.8], label="", color=:black)
		return p
	end
end

# ╔═╡ 672de7fd-1ab5-45e0-a489-7400a11b35e8
begin
	start!(env)
	plot(env)
end


# ╔═╡ 735b43ca-7b9d-4914-8a58-922fb2d9f53d

function main_experiment(parsed=default_args(0.3); progress=false, working=false)

    num_steps = parsed["steps"]
    Random.seed!(parsed["seed"])

    cumulant_schedule = ODTMU.get_cumulant_schedule(parsed)

    # exploring_starts = parsed["exploring_starts"]
    env = OneDTMaze(cumulant_schedule, parsed["exploring_starts"], parsed["env_step_penalty"])

    agent = OneDTmazeExperiment.construct_agent(parsed)

    goal_visitations = zeros(4)

    logger_init_dict = Dict(
        LoggerInitKey.TOTAL_STEPS => num_steps,
        LoggerInitKey.INTERVAL => parsed["logger_interval"],
        # LoggerInitKey.ENV => "tabular_tmaze"
    )
	anim = Animation()


    Curiosity.experiment_wrapper(parsed, logger_init_dict, working) do parsed, logger
        eps = 1
        max_num_steps = num_steps
        steps = Int[]

        logger_start!(logger, env, agent)


        while sum(steps) < max_num_steps
            cur_step = 0
            max_episode_steps = max_num_steps - sum(steps)
            tr, stp =
                run_episode!(env, agent, max_episode_steps) do (s, a, s_next, r, t)
     
					p = plot_env(env)
                    if progress
                        next!(prg_bar)
                    end
					frame(anim)

                    logger_step!(logger, env, agent, s, a, s_next, r, t)
                    cur_step+=1
                end
                logger_episode_end!(logger)
            push!(steps, stp)
            eps += 1
        end
        if working == true
            println("Goal Visitation: ", goal_visitations)
        end
        
    end
	anim

end

# ╔═╡ 7cb251ef-465e-423c-9daa-4aff931c7eb9
anim = main_experiment()

# ╔═╡ 9d4f91d5-e4bb-44b1-8c9a-813beebd0f95
anim

# ╔═╡ Cell order:
# ╠═cf8d2544-a693-11eb-298c-1109f75c63d2
# ╠═21c4a049-a940-4a37-8b6b-a3e09588e98c
# ╠═a32cdc08-b633-41d2-9e2a-555ea9442c49
# ╠═dc8ae3fb-7aad-4449-a7de-55b22a9bb110
# ╠═fce35f63-e4b0-4d4f-b287-936ad5dc4f5e
# ╠═ce948feb-dfff-4337-8f1f-ce72b4028e54
# ╠═672de7fd-1ab5-45e0-a489-7400a11b35e8
# ╠═735b43ca-7b9d-4914-8a58-922fb2d9f53d
# ╠═7cb251ef-465e-423c-9daa-4aff931c7eb9
# ╠═9d4f91d5-e4bb-44b1-8c9a-813beebd0f95
