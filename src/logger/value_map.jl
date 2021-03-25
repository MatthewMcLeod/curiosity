using JLD2
using Statistics
using SparseArrays

mutable struct ValueMap <: LoggerKeyData
    #(behaviour & GVF)x steps x rows x col x actions
    state_action_set::Dict
    value::Array{Float64,3}
    log_interval::Int

    function ValueMap(logger_init_info)
        num_logged_steps = fld(logger_init_info[LoggerInitKey.TOTAL_STEPS], logger_init_info[LoggerInitKey.INTERVAL])
        if logger_init_info["ENV"] == "tabular_tmaze"
            @load string(pwd(),"/src/data/TTMazeValueSet.jld2") ValueSet
        else
            throw(DomainError("Not a valid env setting for logger"))
        end

        state_action_set = ValueSet

        num_gvfs = 4
        num_behaviour  = 1
        value = zeros(num_logged_steps, length(state_action_set["actions"]), num_gvfs + num_behaviour)
        new(state_action_set, value, logger_init_info[LoggerInitKey.INTERVAL])
    end
end

function lg_step!(self::ValueMap, env, agent, s, a, s_next, r, is_terminal, cur_step_in_episode, cur_step_total)
    if rem(cur_step_total, self.log_interval) == 0
        ind = fld(cur_step_total, self.log_interval)
        for (state_ind,(state,action)) in enumerate(zip(self.state_action_set["states"], self.state_action_set["actions"]))
            self.value[ind,state_ind,1] = predict(agent.behaviour_learner, agent, agent.behaviour_weights, state, action)
        end
        for (state_ind,(state,action)) in enumerate(zip(self.state_action_set["states"], self.state_action_set["actions"]))
            self.value[ind,state_ind,2:end] = predict(agent.demon_learner, agent, agent.demon_weights, state, action)
        end
    end
end

function lg_episode_end!(self::ValueMap, cur_step_in_episode, cur_step_total)
end

function save_log(self::ValueMap, save_dict::Dict)
    save_dict[:value_map] = self.value
end
