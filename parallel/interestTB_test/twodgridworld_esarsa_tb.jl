#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o twodgridworld_esarsa_tb.out # Standard output
#SBATCH -e twodgridworld_esarsa_tb.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 3 GB
#SBATCH --time=03:00:00 #
#SBATCH --ntasks=128
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/interestTB_test/twodgridworld_esarsa_tb.toml", save_path="~/scratch/curiosity")
