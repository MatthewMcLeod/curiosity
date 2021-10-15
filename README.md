# Continual Auxiliary Task Learning


This repo contains the code for reproducing the results published in **Continual Auxiliary Task Learning**. This paper was published in NeurIPS2021.

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

### Reproducing results

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
