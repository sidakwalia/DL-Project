# For running a node for two hours
sbatch --time=02:00:00 --mem=1GB --wrap "sleep infinity";
squeue -u $USER;