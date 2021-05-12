#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.6.0/bin/julia
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o emphatic_control.out # Standard output
#SBATCH -e emphatic_control.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 3 GB
#SBATCH --time=00:30:00 #
#SBATCH --ntasks=16
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/interestTB_test/experiment_4_gpi.toml", save_path="~/scratch/curiosity")
