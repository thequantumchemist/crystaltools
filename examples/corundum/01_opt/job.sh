#!/bin/bash
 
#SBATCH --job-name=corundum
#SBATCH --partition=mpi,normal          # select queue (partition), could be mpi, normal or fat
#SBATCH --nodes=1                # request minimum of two nodes
#SBATCH --ntasks-per-node=64     # request 64 mpi ranks per node
#SBATCH --cpus-per-task=1        # request one core per rank, so 1*64 cores total
#SBATCH --mem-per-cpu=6400M
#SBATCH --time=12:00:00          # wallclock time, default wallclock time is 0 !
##SBATCH --gres=ramdisk:25       # request 25 GB memory as local scratch this is /ramdisk

module load orca
export PYTHONPATH=/home1/wido/sw/envs/globalopt/lib/python3.9/site-packages/
source /home1/wido/sw/envs/globalopt/bin/activate
source ~/sw/crystal/crystal23_v1_0_1_Linux-ifort21.4_openmpi4.1.1_exe/utils23/cry23.bashrc
#export CRY23_SCRDIR=/ramdisk/$SLURM_JOBID

#python transform.py sio2_1_ml.traj
#cp sio2.34 sio2.gui

~/sw/crystal/crystal23_v1_0_1_Linux-ifort21.4_openmpi4.1.1_exe/utils23/runPcry23 64 struc


#cp sio2.gui sio2.34
#python retransform.py sio2.34
