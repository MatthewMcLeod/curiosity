
module MCPolicyPlotting
using Plots

using Curiosity

actions = [1,2,3]

function load_policy(policy_name)
    Curiosity.LearnedPolicy("/home/matthewmcleod/Documents/Masters/curiosity/src/data/MC_learned_policies/$(policy_name).bson")
end

function get_q(LP,state_space)
    q = zeros(length(actions),length(state_space),length(state_space))
    for a in actions
        for (i,x) in enumerate(state_space)
            for (j,y) in enumerate(state_space)
                q[a,i,j] = LP([x,y])[a]
            end
        end
    end
    return q
end

function plot_hms(qs,state_space)
    hms = []
    for a in actions
        hm = heatmap(state_space,state_space,qs[a,:,:],xlabel="position", ylabel="normed velocity", title = "Action $(a)",figsize=(1000,1000))
        push!(hms,hm)
    end
    p = plot(hms...)
    return p
end

function load_and_plot_hms(policy_name)
    LP = load_policy(policy_name)
    state_space = collect(0:0.01:1)
    qs = get_q(LP,state_space)
    p = plot_hms(qs, state_space)
    display(p)
    savefig("./plotting/plots/tmp/$(policy_name).png")
end



#
# LP = load_policy("Wall")
# state_space = collect(0:0.01:1)
#
# ps = []
# hms = []
# for a in actions
#     # p = plot(state_space,state_space,q[a,:,:],st=:surface, xlabel="position", ylabel="normed velocity", camera=(-30,30), title = "Action $(a)")
#     # push!(ps,p)
#     hm = heatmap(state_space,state_space,q[a,:,:],xlabel="position", ylabel="normed velocity", title = "Action $(a)",figsize=(1000,1000))
#     push!(hms,hm)
# end
#
# plot(hms..., figsize=(1000,1000))
end
