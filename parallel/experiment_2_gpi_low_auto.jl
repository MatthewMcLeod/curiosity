#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.5.2/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o experiment_2_low_gpi.out # Standard output
#SBATCH -e experiment_2_low_gpi.err # Standard error
#SBATCH --mem-per-cpu=3000M # Memory request of 3GB
#SBATCH --time=01:40:00 #
#SBATCH --ntasks=64
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/experiment_2/experiment_2_gpi.toml", save_path="~/scratch/curiosity")
