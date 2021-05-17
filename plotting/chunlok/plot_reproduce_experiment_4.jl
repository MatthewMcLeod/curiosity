using ProgressMeter

include("./plot_ic_reward.jl")
include("./plot_ic_goal_visitation.jl")

# Experiment parameters
log_interval = 100
num_steps = 60000

# Getting baseline TB
folder = "M:/globus/InterestTB/OneDTMaze_Control_dpi/"
ic = ItemCollection(folder)

# println(diff(ic))

search_ic = search(ic, Dict("demon_learner" => "SR", "demon_update" => "TB", "behaviour_learner" => "GPI"))
gpi_tb_ic = get_best(search_ic, ["eta"], :oned_tmaze_dpi_error)

search_ic = search(ic, Dict("demon_learner" => "SR", "demon_update" => "TB", "behaviour_learner" => "Q"))
esarsa_tb_ic = get_best(search_ic, ["eta"], :oned_tmaze_dpi_error)


configs = [
    Dict(
        :search => Dict("demon_learner" => "Q", "demon_update" => "ETB"),
        :label => "Q-ETB"
    )
    Dict(
        :search => Dict("demon_learner" => "Q", "demon_update" => "EmphESARSA"),
        :label => "Q-EmphESARSA"
    )
    Dict(
        :search => Dict("demon_learner" => "Q", "demon_update" => "InterestTB"),
        :label => "Q-InterestTB"
    )
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

# # # Plotting sum reward for all GVF for ESARSA
# folder = "M:/globus/InterestTB/Experiment4_ESARSA_neg/"
# ic = ItemCollection(folder)
# # println(diff(ic))

# p = plot()
# @showprogress for config in configs
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["alpha_init", "eta"], :oned_tmaze_dpi_error)
#     plot_ic_reward!(p, best_ic, :oned_tmaze_dpi_error, log_interval; smooth_step=1, label=config[:label])
# end

# plot_ic_reward!(p, esarsa_tb_ic, :oned_tmaze_dpi_error, log_interval; smooth_step=1, label="SR TB")
# plot!(title="Q")

# savefig("plotting/chunlok/generated_plots/Experiment4_rewards_esarsa.svg")

# # Plotting sum reward for all GVF for GPI
folder = "M:/globus/InterestTB/Experiment4_GPI_neg/"
ic = ItemCollection(folder)
# # println(diff(ic))

p = plot()
for config in configs
    local search_ic = search(ic, config[:search])
    local best_ic = get_best(search_ic, ["alpha_init", "eta"], :oned_tmaze_dpi_error)
    println("$(config[:search]["demon_learner"]) $(config[:search]["demon_update"])")
    print_params(best_ic, ["alpha_init", "eta"], [])
    plot_ic_reward!(p, best_ic, :oned_tmaze_dpi_error, log_interval; smooth_step=1, label=config[:label])
end

plot_ic_reward!(p, gpi_tb_ic, :oned_tmaze_dpi_error, log_interval; smooth_step=1, label="SR TB")
plot!(title="GPI")
savefig("plotting/chunlok/generated_plots/Experiment4_rewards_GPI.svg")



# Plotting reward per GVF for GPI
# best_ics = []
# for config in configs
#     local search_ic = search(ic, config[:search])
#     local best_ic = get_best(search_ic, ["alpha_init", "eta"], :oned_tmaze_dpi_error)
#     push!(best_ics, best_ic)
# end


# for gvf_i in 1:4
#     p = plot()
#     for (config_i, config) in enumerate(configs)
#         plot_ic_reward_per_gvf!(p, best_ics[config_i], :oned_tmaze_dpi_error, log_interval, gvf_i; smooth_step=1, label=config[:label])
#     end

#     plot_ic_reward_per_gvf!(p, gpi_tb_ic, :oned_tmaze_dpi_error, log_interval, gvf_i; smooth_step=1, label="SR TB")

#     plot!(title="GVF $(gvf_i)")
#     savefig("plotting/chunlok/generated_plots/Experiment4_rewards_GPI_gvf_$(gvf_i).svg")
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




