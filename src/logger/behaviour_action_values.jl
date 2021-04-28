
mutable struct BehaviourActionValues <: LoggerKeyData
    max_q::Array{Float64}
    function BehaviourActionValues(logger_init_info)
        new([])
    end
end

function lg_step!(self::BehaviourActionValues, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    base_s = proc_input(agent, s)
    q = maximum(agent.behaviour_learner(base_s))
    push!(self.max_q,q)
end

function lg_episode_end!(self::BehaviourActionValues, cur_step_in_episode, cur_step_total)
end

function save_log(self::BehaviourActionValues, save_dict::Dict)
    save_dict[:max_behaviour_q] = self.max_q
end
