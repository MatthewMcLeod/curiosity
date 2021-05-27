#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o twodgridworld_gpi.out # Standard output
#SBATCH -e twodgridworld_gpi.err # Standard error
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=48
#SBATCH --mem=0
#SBATCH --time=08:00:00 #
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/interestTB_test/twodgridworld_gpi.toml", save_path="~/scratch/curiosity")
