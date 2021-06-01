using Plots; pyplot()
using Reproduce
using FileIO
using Statistics
using ProgressMeter
using JLD2
# import ..GeneralPlotUtils
# import ..LabelUtils
# include("./plot_utils.jl")
include("./plot_utils.jl")

GPU = GeneralPlotUtils
LU = LabelUtils


data_home
data_key = :twod_grid_world_error_center_dpi
# data_key = :oned_tmaze_dmu_error


folder_name = "twod_gpi"
data_home = "../TwoDGridWorld_gpi_StepSizes"

function load_data()
    experiment_folders = [data_home]
    # folder_name = "oned_rr_dpi"
    # folder_name = "oned_control_dpi"
    ic = ItemCollection(joinpath(experiment_folders[1], "data"))
    return ic
end

function load_best(ic)
# data_key = :oned_tmaze_dpi_error

    algo_divisor_keys = ["behaviour_learner", "demon_learner", "demon_opt", "demon_update"]
    sweep_params = ["demon_alpha_init", "demon_eta", "alpha_init"]

    algo_specs_full = GPU.split_algo(ic, algo_divisor_keys)

    all_algos_ics = [search(ic,algo_spec) for algo_spec in algo_specs_full]
    @show length.(all_algos_ics)

    valid_algos_ind = findall(a -> length(a) != 0, all_algos_ics)

    algo_specs = algo_specs_full[valid_algos_ind]
    algo_ics = all_algos_ics[valid_algos_ind]
    best_per_algo_ics = []
    for (i,algo_ic) in enumerate(algo_ics)
        push!(best_per_algo_ics, GPU.get_best_final_perf(algo_ic,sweep_params, data_key, 0.1))
    end
    @show length.(best_per_algo_ics)
    return best_per_algo_ics
end

ic = load_data()
best_ics = load_best(ic)

stepsizes = [GPU.load_results(algo_ic,:autostep_stepsize; return_type = "array") for algo_ic in best_ics]
#algo x runs x steps x [SF, params]

ss_algos = []
ss_algos_SF = []
for algo_ss in stepsizes
    ss_algo = []
    ss_algo_SF = []
    for run_ss in algo_ss
        # @show size(run_ss)
        # @show size(run_ss[1])
        # @show size(run_ss[1][1])
        r = []
        SF = []
        for step in run_ss
            if size(step[1]) == (4,4)
                push!(r, step[1])
                push!(SF,step[2])
            else
                push!(r,step[2])
                push!(SF,step[1])
            end
            # @show size(r[end])
        end
        push!(ss_algo, cat(r..., dims=3))
        push!(ss_algo_SF, cat(SF..., dims=3))
        # tmp_r = cat([step[1] for step in run_ss]..., dims = 3)
        # tmp_sf = cat([step[2] for step in run_ss]...,dims=3)

        # @show size(tmp_r)
        # push!(ss_algo, tmp_r)
    end
    # @show size(cat(ss_algo..., dims = 4))
    push!(ss_algos,cat(ss_algo..., dims = 4))
    push!(ss_algos_SF,cat(ss_algo_SF..., dims = 4))

end
function extract_cumulants_ss(ss_algo)
    distractor = ss_algo[1,1,:,:]
    drifter = ss_algo[3,3,:,:]
    c1 = ss_algo[2,2,:,:]
    c2 = ss_algo[4,4,:,:]
    return distractor,c1,drifter,c2
end

ps = [plot(title="Distractor"),plot(title="Constant"),plot(title="Drifter"),plot(title="Constant")]
ps_SF = [plot(title="Distractor"),plot(title="Constant"),plot(title="Drifter"),plot(title="Constant")]

for (ind,ss_algo) in enumerate(ss_algos)
    demon_update = best_ics[ind].items[1].parsed_args["demon_update"]
    dis,c1,drif,c2 = extract_cumulants_ss(ss_algo)

    #Rewards
    plot!(ps[1],mean(dis,dims=2), label = demon_update, legend=:topleft)
    plot!(ps[2],mean(c1,dims=2), label = demon_update, legend=:topleft)
    plot!(ps[3],mean(drif,dims=2), label = demon_update, legend=:topleft)
    plot!(ps[4],mean(c2,dims=2), label = demon_update, legend=:topleft)

    #SF
    @show size(ss_algos_SF[ind])
    dis_SF = mean(ss_algos_SF[ind][1:4,:,:,:], dims = [1,2,4])
    c1_SF = mean(ss_algos_SF[ind][5:8,:,:,:], dims = [1,2,4])
    drif_SF = mean(ss_algos_SF[ind][9:12,:,:,:], dims = [1,2,4])
    c2_SF = mean(ss_algos_SF[ind][13:16,:,:,:], dims = [1,2,4])

    @show mean(dis_SF)
    @show mean(c1_SF)
    @show mean(drif_SF)
    @show mean(c2_SF)

    @show vec(drif_SF)[end]

    plot!(ps_SF[1],vec(dis_SF), label = demon_update, legend=:topleft)
    plot!(ps_SF[2], vec(c1_SF), label = demon_update, legend=:topleft)
    plot!(ps_SF[3], vec(drif_SF), label = demon_update, legend=:topleft)
    plot!(ps_SF[4], vec(c2_SF), label = demon_update, legend=:topleft)

end
display(plot(ps...))
display(plot(ps_SF...))
