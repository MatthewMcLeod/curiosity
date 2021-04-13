#!/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/julia/1.5.2/bin/julia
#SBATCH --mail-user=chunlok@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH -o experiment_2_gpi.out # Standard output
#SBATCH -e experiment_2_gpi.err # Standard error
#SBATCH --mem-per-cpu=4000M # Memory request of 3 GB
#SBATCH --time=00:20:00 #
#SBATCH --ntasks=128
#SBATCH --account=def-whitem

using Pkg
Pkg.activate(".")

include(joinpath(ENV["SLURM_SUBMIT_DIR"], "parallel/parallel_config.jl"))
reproduce_config_experiment("configs/experiment_2_gpi.toml", save_path="~/scratch/curiosity")
