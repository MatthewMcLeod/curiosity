#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o onedtmaze_esarsa.out # Standard output
#SBATCH -e onedtmaze_esarsa.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 3GB
#SBATCH --time=02:00:00 #
#SBATCH --ntasks=32
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/onedtmaze_control_dpi/ESARSA_Auto.toml", save_path="~/scratch/curiosity")
