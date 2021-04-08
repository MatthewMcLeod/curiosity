#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.5.2/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o experiment_1_part_1.out # Standard output
#SBATCH -e experiment_1_part_1.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 2 GB
#SBATCH --time=02:00:00 #
#SBATCH --ntasks=6
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/experiment_1_part_1.toml", save_path="~/scratch/curiosity")
