#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o onedtmaze_GPI_tilings.out # Standard output
#SBATCH -e onedtmaze_GPI_tilings.err # Standard error
#SBATCH --mem-per-cpu=7000M # Memory request of 3GB
#SBATCH --time=03:30:00 #
#SBATCH --ntasks=1
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/onedtmaze/GPI_tilecode_features_large_tilings.toml", save_path="~/scratch/curiosity")
