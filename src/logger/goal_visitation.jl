
mutable struct GoalVisitation <: LoggerKeyData
    goal_visitations::Array{Int64}
    
    function GoalVisitation()
        new(ones(4))
    end
end

function step!(self::GoalVisitation, env, agent, s, a, s_next, r)
    C,_,_ = get(agent.demons, s, a, s_next)
    if sum(C) != 0
        gvf_i = findfirst(!iszero,C)
        self.goal_visitations[gvf_i] += 1
    end
end

function save_log(self::GoalVisitation, save_dict::Dict)
    per = [self.goal_visitations[i] / sum(self.goal_visitations) for i in 1:4]
    println(self.goal_visitations)
    save_dict[:goal_visitation] = per
end