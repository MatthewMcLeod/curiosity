
mutable struct GoalVisitation <: LoggerKeyData
    goal_visitations::Array{Int64}

    function GoalVisitation(logger_init_info)
        new(ones(4))
    end
end

function lg_step!(self::GoalVisitation, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    # C,_,_ = get(agent.demons, s, a, s_next)
    #NOTE This only works for TTMaze. 
    C = s_next[2:5]
    if sum(C) != 0
        gvf_i = findfirst(!iszero,C)
        self.goal_visitations[gvf_i] += 1
    end
end

function lg_episode_end!(self::GoalVisitation, cur_step_in_episode, cur_step_total)
end

function save_log(self::GoalVisitation, save_dict::Dict)
    save_dict[:goal_visitation] = self.goal_visitations
end
