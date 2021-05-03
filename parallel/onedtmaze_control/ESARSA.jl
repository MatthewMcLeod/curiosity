#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.5.2/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o onedtmaze_ESARSA_Control.out # Standard output
#SBATCH -e onedtmaze_ESARSA_Control.err # Standard error
#SBATCH --mem-per-cpu=6000M # Memory request of 3GB
#SBATCH --time=02:00:00 #
#SBATCH --ntasks=16
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/onedtmaze_control/ESARSA_behaviour.toml", save_path="~/scratch/curiosity")
