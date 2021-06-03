include("buffer.jl")

import Random
import DataStructures
import MacroTools: @forward
import StatsBase

abstract type AbstractReplay end

proc_state(er::AbstractReplay, x) = identity(x)
Base.keys(er::AbstractReplay) = 1:length(er)

function Base.iterate(asr::AbstractReplay)

    state = 1
    result = asr[state]
    state += 1
    (result, state)
end

function Base.iterate(asr::AbstractReplay, state::Integer)

    if state > length(asr)
        return nothing
    end
    
    result = asr[state]
    state += 1
    
    (result, state)
end

mutable struct ExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
end

const ExperienceReplayDef{TPL, TPS} = ExperienceReplay{CircularBuffer{TPL, TPS, (:s, :a, :sp, :r, :t)}}

ExperienceReplay(size, types, shapes, column_names) = begin
    cb = CircularBuffer(size, types, shapes, column_names)
    ExperienceReplay(cb)
end

ExperienceReplayDef(size, obs_size, obs_type) =
    ExperienceReplay(size,
                     (obs_type, Int, obs_type, Float32, Bool),
                     (obs_size, 1, obs_size, 1, 1),
                     (:s, :a, :sp, :r, :t))

Base.length(er::ExperienceReplay) = length(er.buffer)
Base.getindex(er::ExperienceReplay, idx) = er.buffer[idx]
Base.view(er::ExperienceReplay, idx) = @view er.buffer[idx]

Base.push!(er::ExperienceReplay, experience) = push!(er.buffer, experience)

StatsBase.sample(er::ExperienceReplay, batch_size) = StatsBase.sample(Random.GLOBAL_RNG, er, batch_size)

function StatsBase.sample(rng::Random.AbstractRNG, er::ExperienceReplay, batch_size)
    idx = rand(rng, 1:length(er), batch_size)
    return er[idx]
end

mutable struct DynaExperienceReplay{CB<:CircularBuffer} <: AbstractReplay
    buffer::CB
end

DynaExperienceReplay(size, types, shapes, column_names) = begin
    cb = CircularBuffer(size, types, shapes, column_names)
    DynaExperienceReplay(cb)
end

Base.length(er::DynaExperienceReplay) = length(er.buffer)
Base.getindex(er::DynaExperienceReplay, idx) = er.buffer[idx]
Base.view(er::DynaExperienceReplay, idx) = @view er.buffer[idx]

Base.push!(er::DynaExperienceReplay, experience) = push!(er.buffer, experience)

StatsBase.sample(er::DynaExperienceReplay, batch_size) = StatsBase.sample(Random.GLOBAL_RNG, er, batch_size)

function StatsBase.sample(rng::Random.AbstractRNG, er::DynaExperienceReplay, batch_size)
    idxes = rand(rng, 1:length(er), batch_size)
    [er[idx] for idx in idxes]
end
