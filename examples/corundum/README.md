# cryfreq test: alpha-Al2O3 (corundum), HSE06 / pob-TZVP-rev2

Once on the cluster:  pip install --user phonopy

1. Optimization
       cd 01_opt && sbatch job.sh
2. Classical reference (FREQCALC + INTENS, Berry phase), after step 1 has finished.
   The job starts Pcrystal directly so that BORN.DAT (CRYSTAL's Born charges) is kept.
       cp 01_opt/struc.gui 02_freq_classic/ && cd 02_freq_classic && sbatch job.sh
3. Automated route (can run at the same time as step 2)
       cd 03_cryfreq
       python ../../../cryfreq.py setup --geom ../01_opt/struc.out --input ../02_freq_classic/struc.d12 --cores 64 --time 02:00:00
       sbatch run_array.sh            # forces (GRADCAL) + POLARI for ref and every displacement
       python ../../../cryfreq.py status
       sbatch run_spolbp.sh           # after the array has finished: SPOLBP for every displacement
4. Comparison
       python ../../../cryfreq.py analyze --polar --compare ../02_freq_classic/struc.out

Results: frequencies.dat, born_charges.dat, ir_spectrum.dat/.png and a mode-by-mode comparison
with FREQCALC (frequencies, irreps, IR intensities) on screen.
Please send back: 02_freq_classic/struc.out, 02_freq_classic/BORN.DAT, 03_cryfreq/disp-001/spolbp.outp,
and the screen output of step 4.
