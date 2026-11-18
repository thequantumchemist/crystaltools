#!/bin/bash
 
#SBATCH --job-name=corundum_freq
#SBATCH --partition=mpi,normal          # select queue (partition), could be mpi, normal or fat
#SBATCH --nodes=1                # request minimum of two nodes
#SBATCH --ntasks-per-node=64     # request 64 mpi ranks per node
#SBATCH --cpus-per-task=1        # request one core per rank, so 1*64 cores total
#SBATCH --mem-per-cpu=6400M
#SBATCH --time=12:00:00          # wallclock time, default wallclock time is 0 !
##SBATCH --gres=ramdisk:25       # request 25 GB memory as local scratch this is /ramdisk

source ~/sw/crystal/crystal23_v1_0_1_Linux-ifort21.4_openmpi4.1.1_exe/utils23/cry23.bashrc
# Pcrystal is started directly (not via runPcry23) so that BORN.DAT is kept
EXE=${CRY23_EXEDIR}/${VERSION}
MPIRUN=/cluster/mpi/openmpi/4.1.5/gcc-13.1.0/bin/mpirun
rm -rf run && mkdir run && cd run
cp ../struc.d12 INPUT
cp ../struc.gui fort.34
$MPIRUN -np 64 $EXE/Pcrystal > ../struc.out 2>&1
cp BORN.DAT FREQINFO.DAT ../ 2>/dev/null
cd .. && ls -l
