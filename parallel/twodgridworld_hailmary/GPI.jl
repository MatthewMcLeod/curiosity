#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=mmcleod2@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o twod_hail.out # Standard output
#SBATCH -e twod_hail.err # Standard error
#SBATCH --time=02:30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0
#SBATCH --account=def-amw8

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/twodgridworld_hailmary/twodgridworld_gpi.toml", save_path="~/scratch/curiosity")
