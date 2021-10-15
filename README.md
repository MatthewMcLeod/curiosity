# Continual Auxiliary Task Learning


This repo contains the code for reproducing the results published in **Continual Auxiliary Task Learning**. This paper was published in [NeurIPS2021](link).

## Authors
- Matthew McLeod
- Chunlok Lo
- Andrew Jacobsen 
- [Matthew Schlegel](https://mkschleg.github.io)
- Raksha Kumaraswamy
- [Adam White](https://sites.ualberta.ca/~amw8/)
- [Martha White](https://webdocs.cs.ualberta.ca/~whitem/)


## Abstract
Learning auxiliary tasks, such as multiple predictions about the world, can provide many benefits to reinforcement learning systems. A variety of off-policy learning algorithms have been developed to learn such predictions, but as yet there is little work on how to adapt the behavior to gather useful data for those off-policy predictions. In this work, we investigate a reinforcement learning system designed to learn a collection of auxiliary tasks, with a behavior policy learning to take actions to improve those auxiliary predictions. We highlight the inherent non-stationarity in this continual auxiliary task learning problem, for both prediction learners and the behavior learner. We develop an algorithm based on successor features that facilitates tracking under non-stationary rewards, and prove  the separation into learning successor features and rewards provides convergence rate improvements. We conduct an in-depth study into the resulting multi-prediction learning system. 


## Detailed instructions:

If one was able to run the code as is, the `parallel/toml_parallel.jl` file is the entry point. This parallelizes our sweeps and works with all the listed config files.

### Downloading and installing Julia

Julia can be found and downloaded [here](julialang.org). You can also find details on the language in the [documentation](https://docs.julialang.org/en/v1/). We only guarantee this code works for versions up-to v1.5.x. 



### Reproducing results

1. Install julia version 1.5.x from https://julialang.org/downloads/
2. Add to path
3. `cd` to the `curiosity` directory
4. Instantiate the project:

```julia
%> julia --project
julia> ]instantiate
```

5. To run an experiment from its config:

```bash
julia --project parallel/toml_parallel.jl <<config_file.toml>> 
```

This will run the experiment and place the results in the folder defined in the toml. To run on a larger cluster is possible, but requires more details. You should see [Reproduce.jl](https://github.com/mkschleg/Reproduce.jl/blob/master/Project.toml) for an example/details.


#### Config Files Mapped to Experiments:
- Figure 2:
- Figure 3:
- Figure 4:

#### Appendix empirical results:
- Figure 5:
- Figure 6:
- Figure 7:
- Figure 8:

### Analyzing data with [Reproduce.jl](https://github.com/mkschleg/Reproduce.jl/blob/master/Project.toml) and [ReproducePlotUtils.jl](https://github.com/mkschleg/ReproducePlotUtils.jl/tree/master/src).

## Acknoledgements
