#!/bin/bash
#SBATCH --job-name=cryfreq_polar
#SBATCH --partition=mpi,normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6400M
#SBATCH --time=06:00:00
#
# =============================================================================
#  polar_array.sh
#
#  Berry-phase polarization (POLARI + SPOLBP) for every structure of a
#  cryfreq folder, as a SLURM array job.
#
#  Usage - inside a cryfreq folder (the one with phonopy_disp.yaml, ref/, disp-XXX/):
#
#      bash polar_array.sh
#
#  This does NOT run anything itself. It
#    1. writes polari.d3 (NEWK mesh taken from SHRINK of ref/struc.d12) and spolbp.d3,
#    2. submits task 0      : POLARI for ref/
#    3. submits tasks 1..N  : POLARI + SPOLBP for disp-001 ... disp-N,
#       starting only after task 0 has finished successfully,
#       with at most MAX_PARALLEL tasks running at the same time.
#
#  Finished steps are skipped, so the same command can simply be repeated
#  after a time-out. Afterwards:   python cryfreq.py analyze --polar
# =============================================================================


# =============================================================================
#  SETTINGS
# =============================================================================

# number of array tasks running at the same time
MAX_PARALLEL=10

# CRYSTAL environment (defines CRY23_EXEDIR and VERSION)
CRYSTAL_ENV=${CRYFREQ_CRYSTAL_ENV:-~/sw/crystal/crystal23_v1_0_1_Linux-ifort21.4_openmpi4.1.1_exe/utils23/cry23.bashrc}

# input files written into the cryfreq folder
POLARI_INPUT=polari.d3
SPOLBP_INPUT=spolbp.d3

# name of this script (needed when it is submitted by mistake with sbatch)
SCRIPT_NAME=polar_array.sh


# =============================================================================
#  SUBMIT MODE  (anything that is not an array task: "bash polar_array.sh",
#                or by mistake "sbatch polar_array.sh")
# =============================================================================

if [ -z "${SLURM_ARRAY_TASK_ID}" ]; then

    # path of this script - inside a plain sbatch job $0 is SLURM's internal copy
    if [ -n "${SLURM_JOB_ID}" ]; then
        cd ${SLURM_SUBMIT_DIR}
        SCRIPT=${SLURM_SUBMIT_DIR}/${SCRIPT_NAME}
    else
        SCRIPT=$0
    fi

    if [ ! -f "${SCRIPT}" ]; then
        echo "ERROR: ${SCRIPT} not found - start this script with:  bash ${SCRIPT_NAME}"
        exit 1
    fi

    if [ ! -f phonopy_disp.yaml ] || [ ! -d ref ]; then
        echo "ERROR: run this inside a cryfreq folder (phonopy_disp.yaml and ref/ not found)"
        exit 1
    fi

    N=$(ls -d disp-[0-9][0-9][0-9] 2>/dev/null | wc -l)
    if [ "${N}" -eq 0 ]; then
        echo "ERROR: no disp-XXX folders found"
        exit 1
    fi


    # --- wave functions present? ---------------------------------------------

    MISSING=0
    for d in ref disp-[0-9][0-9][0-9]; do
        [ -s "${d}/struc.f9" ] || { echo "missing wave function: ${d}/struc.f9"; MISSING=1; }
    done
    if [ "${MISSING}" -ne 0 ]; then
        echo "ERROR: run (or finish) run_array.sh first"
        exit 1
    fi


    # --- NEWK block = same mesh as SHRINK of the SCF -------------------------

    SHRINK_LINES=$(sed -n '/^[[:space:]]*SHRINK[[:space:]]*$/{n;p;n;p;q}' ref/struc.d12)
    LINE1=$(echo "${SHRINK_LINES}" | sed -n 1p)
    LINE2=$(echo "${SHRINK_LINES}" | sed -n 2p)

    if [ -z "${LINE1}" ]; then
        echo "ERROR: no SHRINK found in ref/struc.d12"
        exit 1
    fi

    if [ "$(echo ${LINE1} | awk '{print $1}')" = "0" ]; then
        NEWK_MESH="${LINE1}\n${LINE2}"
    else
        NEWK_MESH="${LINE1}"
    fi

    printf "NEWK\n${NEWK_MESH}\n0 0\nPOLARI\nEND\n" > ${POLARI_INPUT}
    printf "SPOLBP\nEND\n"                       > ${SPOLBP_INPUT}

    echo "written: ${POLARI_INPUT} (NEWK as SHRINK:  $(echo -e ${NEWK_MESH} | tr '\n' ' '))"
    echo "written: ${SPOLBP_INPUT}"


    # --- submit ---------------------------------------------------------------

    REF_JOB=$(sbatch --parsable --array=0 "${SCRIPT}")
    echo "submitted reference      (task 0):         job ${REF_JOB}"

    DISP_JOB=$(sbatch --parsable --array=1-${N}%${MAX_PARALLEL} \
                      --dependency=afterok:${REF_JOB} "${SCRIPT}")
    echo "submitted displacements  (tasks 1-${N}, ${MAX_PARALLEL} at a time): job ${DISP_JOB}"

    exit 0
fi


# =============================================================================
#  WORKER MODE  (one array task = one structure)
# =============================================================================

source ${CRYSTAL_ENV}
PROP=${CRY23_EXEDIR}/${VERSION}/properties

cd ${SLURM_SUBMIT_DIR}

TASK=${SLURM_ARRAY_TASK_ID}
if [ "${TASK}" -eq 0 ]; then
    DIR=ref
else
    DIR=$(printf "disp-%03d" ${TASK})
fi


# --- step 1: POLARI (preliminary Berry-phase run, unit 27 -> pol.f27) ----------

run_polari () {

    if [ -s "${DIR}/pol.f27" ] && grep -q "FORTRAN UNIT 27" "${DIR}/polari.outp" 2>/dev/null; then
        echo "${DIR}: POLARI already done"
        return 0
    fi

    rm -rf ${DIR}/polrun && mkdir ${DIR}/polrun
    (
        cd ${DIR}/polrun                                   &&
        cp ../struc.f9 fort.9                             &&
        ${PROP} < ../../${POLARI_INPUT} > ../polari.outp 2>&1 &&
        cp fort.27 ../pol.f27
    )
    STATUS=$?
    rm -rf ${DIR}/polrun

    if [ ${STATUS} -ne 0 ] || [ ! -s "${DIR}/pol.f27" ]; then
        echo "${DIR}: POLARI FAILED - see ${DIR}/polari.outp"
        return 1
    fi

    echo "${DIR}: POLARI done"
}


# --- step 2: SPOLBP (polarization displaced - reference) ----------------------

run_spolbp () {

    if grep -q "QUANTUM" "${DIR}/spolbp.outp" 2>/dev/null; then
        echo "${DIR}: SPOLBP already done"
        return 0
    fi

    if [ ! -s ref/pol.f27 ]; then
        echo "${DIR}: ref/pol.f27 missing - reference not finished"
        return 1
    fi

    rm -rf ${DIR}/sprun && mkdir ${DIR}/sprun
    (
        cd ${DIR}/sprun                                    &&
        cp ../struc.f9 fort.9                             &&
        cp ../../ref/pol.f27 fort.28                      &&
        cp ../pol.f27 fort.29                             &&
        ${PROP} < ../../${SPOLBP_INPUT} > ../spolbp.outp 2>&1
    )
    STATUS=$?
    rm -rf ${DIR}/sprun

    if [ ${STATUS} -ne 0 ] || ! grep -q "QUANTUM" "${DIR}/spolbp.outp"; then
        echo "${DIR}: SPOLBP FAILED - see ${DIR}/spolbp.outp"
        return 1
    fi

    echo "${DIR}: SPOLBP done"
}


# --- run ----------------------------------------------------------------------

run_polari || exit 1

if [ "${TASK}" -gt 0 ]; then
    run_spolbp || exit 1
fi
