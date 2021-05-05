#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.5.2/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o onedtmaze_ESARSA_Auto_Part_1.out # Standard output
#SBATCH -e onedtmaze_ESARSA_Auto_Part_1.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 3GB
#SBATCH --time=01:00:00 #
#SBATCH --ntasks=6
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/onedtmaze_control/ESARSA_part_1_Auto_Demons.toml", save_path="~/scratch/curiosity")
