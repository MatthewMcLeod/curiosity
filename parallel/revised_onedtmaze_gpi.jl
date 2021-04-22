#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.5.2/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o onedtmaze_gpi.out # Standard output
#SBATCH -e onedtmaze_gpi.err # Standard error
#SBATCH --mem-per-cpu=3000M # Memory request of 3GB
#SBATCH --time=03:30:00 #105 hours total
#SBATCH --ntasks=32
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/revised_onedtmaze/GPI_behaviour.toml", save_path="~/scratch/curiosity")
