#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o onedtmaze_GPI_dpi.out # Standard output
#SBATCH -e onedtmaze_GPI_dpi.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 3GB
#SBATCH --time=05:00:00 #
#SBATCH --ntasks=64
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/onedtmaze_control_dpi/GPI_Auto.toml", save_path="~/scratch/curiosity")
