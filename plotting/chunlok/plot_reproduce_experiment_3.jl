using ProgressMeter

include("./plot_ic_reward.jl")
include("./plot_ic_goal_visitation.jl")

# Experiment parameters
log_interval = 100
num_steps = 40000

# Getting baseline TB
folder = "M:/globus/InterestTB/OneDTMaze_RR_dpi/"
ic = ItemCollection(folder)

search_ic = search(ic, Dict("demon_learner" => "SR", "demon_opt" => "Auto"))
tb_ic = get_best(search_ic, ["demon_eta"], :oned_tmaze_dpi_error)


configs = [
    # Dict(
    #     :search => Dict("demon_learner" => "Q", "demon_update" => "ETB"),
    #     :label => "Q-ETB"
    # )
    # Dict(
    #     :search => Dict("demon_learner" => "Q", "demon_update" => "EmphESARSA"),
    #     :label => "Q-EmphESARSA"
    # )
    # Dict(
    #     :search => Dict("demon_learner" => "Q", "demon_update" => "InterestTB"),
    #     :label => "Q-InterestTB"
    # )
    Dict(
        :search => Dict("demon_learner" => "SR", "demon_update" => "ETB"),
        :label => "SR-ETB"
    )
    Dict(
        :search => Dict("demon_learner" => "SR", "demon_update" => "EmphESARSA"),
        :label => "SR-EmphESARSA"
    )
    Dict(
        :search => Dict("demon_learner" => "SR", "demon_update" => "InterestTB"),
        :label => "SR-InterestTB"
    )
]

folder = "M:/globus/InterestTB/Experiment3/"
ic = ItemCollection(folder)

# println(diff(ic))

# asdfsadf

# Plotting sum reward for all GVF
p = plot()
@showprogress for config in configs
    local search_ic = search(ic, config[:search])
    local best_ic = get_best(search_ic, ["alpha_init", "eta"], :oned_tmaze_dpi_error)
    plot_ic_reward!(p, best_ic, :oned_tmaze_dpi_error, log_interval; smooth_step=10, label=config[:label])
end

ylims!(p, (0, 1.5))
plot_ic_reward!(p, tb_ic, :oned_tmaze_dpi_error, log_interval; smooth_step=10, label="SR TB")

savefig("plotting/chunlok/generated_plots/Experiment3_rewards.svg")

# Plotting reward per GVF
# best_ics = []
# for config in configs
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["alpha_init", "eta"], :oned_tmaze_dpi_error)
#     push!(best_ics, best_ic)
# end


# for gvf_i in 1:4
#     p = plot()
#     for (config_i, config) in enumerate(configs)
#         plot_ic_reward_per_gvf!(p, best_ics[config_i], :oned_tmaze_dpi_error, log_interval, num_steps, gvf_i; smooth_step=1, label=config[:label])
#     end
#     plot!(title="GVF $(gvf_i)")
#     savefig("plotting/chunlok/generated_plots/Experiment4_rewards_gvf_$(gvf_i)")
# end


# Plotting sum reward for all GVF

# for config in configs
#     p = plot()
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["alpha_init", "eta"], :oned_tmaze_dpi_error)
#     plot_ic_goal_visitation!(p, best_ic)
#     plot!(title=config[:label])
#     savefig("plotting/chunlok/generated_plots/Experiment4_goal_visitation_$(config[:label])")
# end




