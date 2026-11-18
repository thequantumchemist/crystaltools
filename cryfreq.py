#!/usr/bin/env python3
"""
cryfreq.py - harmonic Gamma-point frequencies and IR intensities with CRYSTAL + phonopy

Every displaced structure is an independent CRYSTAL single point with forces (GRADCAL),
so no long FREQCALC run is needed. Force constants, frequencies, eigenvectors and
irreducible representations are obtained with phonopy; IR intensities from Born
effective charges (either given directly or derived from cell-dipole changes).

Sub-commands
  setup    create disp-XXX/ (and ref/) folders with struc.d12 + struc.gui and an array job
  status   show which displacements are finished
  analyze  collect forces, compute frequencies (+ IR intensities), compare with FREQCALC

Typical use
  cryfreq.py setup   --geom ../01_opt/struc.out --input ../02_freq_classic/struc.d12
  sbatch run_array.sh
  cryfreq.py status
  sbatch run_spolbp.sh                      (after the array has finished)
  cryfreq.py analyze --polar --compare ../02_freq_classic/struc.out
  cryfreq.py thermo [--mesh 4 4 6]          ZPE, F_vib(T), S_vib(T) -> thermo.dat
  cryfreq.py compare dirA dirB --at 0 300 1100   differences A - B (electronic + vibrational)
"""
import argparse, os, re, sys, glob, warnings
import numpy as np
warnings.filterwarnings('ignore', category=DeprecationWarning)

HARTREE_BOHR_TO_EV_ANG = 51.42208619083232
THZ_TO_CM = 33.35640951981521
E2_AMU_TO_KMMOL = 974.8801          # IR intensity: (e^2/amu) -> km/mol
# isotopic masses as used by CRYSTAL (most abundant isotope)
MASSES = {'H':1.00783,'Li':7.01600,'B':11.00931,'C':12.0,'N':14.00307,'O':15.99491,'F':18.99840,
          'Na':22.98977,'Mg':23.98504,'Al':26.98154,'Si':27.97693,'P':30.97376,'S':31.97207,
          'Ca':39.96259,'Ti':47.94795,'V':50.94396,'Cr':51.94051,'Mn':54.93805,'Fe':55.93494,
          'Co':58.93320,'Ni':57.93535,'Cu':62.92960,'Zn':63.92915,'Ga':68.92558,'Bi':208.98040}
OPEN_BLOCKS = {'OPTGEOM','FREQCALC','ANHARM','ELASTCON','EOS','IRSPEC','RAMSPEC','PREOPTGEOM','SCANMODE'}

SYMBOLS = ['X','H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
           'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr','Y','Zr',
           'Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba','La','Ce','Pr','Nd',
           'Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
           'Tl','Pb','Bi']
def element(conv):
    return SYMBOLS[conv % 100]          # CRYSTAL convention: 283 = Bi with ECP, 13 = Al all-electron

# ---------------------------------------------------------------- CRYSTAL parsing
def read_crystal_geometry(fname):
    """Last 'DIRECT LATTICE VECTORS CARTESIAN COMPONENTS' + 'CARTESIAN COORDINATES - PRIMITIVE CELL'."""
    t = open(fname).read().split('\n')
    il = [i for i,l in enumerate(t) if 'DIRECT LATTICE VECTORS CARTESIAN COMPONENTS (ANGSTROM)' in l]
    ic = [i for i,l in enumerate(t) if 'CARTESIAN COORDINATES - PRIMITIVE CELL' in l]
    if not il or not ic: sys.exit(f'no geometry found in {fname}')
    i = il[-1]; lat = np.array([list(map(float, t[i+2+k].split())) for k in range(3)])
    j = [c for c in ic if c > i]; j = j[0] if j else ic[-1]
    k = j + 4; conv = []; pos = []
    while k < len(t) and t[k].strip() and re.match(r'^\s*\d+\s+\d+\s+[A-Z]', t[k]):
        p = t[k].split(); conv.append(int(p[1])); pos.append(list(map(float, p[3:6]))); k += 1
    return lat, np.array(conv), np.array(pos)

def split_input(fname):
    """Return the part of a CRYSTAL input that follows the geometry block, without the final END."""
    lines = open(fname).read().split('\n')
    depth = 0; gend = None
    for i, l in enumerate(lines[1:], start=1):
        w = l.strip().upper()
        if w in OPEN_BLOCKS: depth += 1
        elif w.startswith('END'):
            if depth > 0: depth -= 1
            else: gend = i; break
    if gend is None: sys.exit('could not find the end of the geometry block in ' + fname)
    rest = lines[gend+1:]
    while rest and not rest[-1].strip(): rest.pop()
    if not rest[-1].strip().upper().startswith('END'): sys.exit('input does not end with END')
    rest = rest[:-1]
    if any(l.strip().upper() == 'GRADCAL' for l in rest): rest = [l for l in rest if l.strip().upper() != 'GRADCAL']
    return '\n'.join(rest)

def read_forces(fname, nat):
    """Last 'CARTESIAN FORCES IN HARTREE/BOHR' block -> eV/Angstrom."""
    t = open(fname).read().split('\n')
    idx = [i for i,l in enumerate(t) if 'CARTESIAN FORCES IN HARTREE/BOHR' in l]
    if not idx: return None
    i = idx[-1] + 2; f = []
    for k in range(nat):
        p = t[i+k].split(); f.append(list(map(float, p[2:5])))
    return np.array(f) * HARTREE_BOHR_TO_EV_ANG

def read_freqcalc(fname):
    """Mode table of a classical FREQCALC output (first table only)."""
    rows = []; started = False
    pat = re.compile(r'^\s*(\d+)-\s*(\d+)\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\(\s*(\S+)\s*\)\s+([AI])\s+\(\s*([-\d.]+)\s*\)')
    for l in open(fname):
        m = pat.match(l)
        if m:
            started = True
            a, b = int(m.group(1)), int(m.group(2))
            rows.append(dict(first=a, last=b, freq=float(m.group(4)), irrep=m.group(6),
                             ir=m.group(7), intens=float(m.group(8))))
        elif started and rows and not l.strip(): break
    return rows

# ---------------------------------------------------------------- helpers
def make_cell(lat, conv, pos):
    from phonopy.structure.atoms import PhonopyAtoms
    sym = [element(c) for c in conv]
    return PhonopyAtoms(symbols=sym, cell=lat, positions=pos, masses=[MASSES.get(s) for s in sym])

def newk_from_template(template):
    """NEWK block with the same mesh as SHRINK in the template, IFE=0 (as required for POLARI)."""
    l = [x.strip() for x in template.split('\n')]
    i = [k for k, x in enumerate(l) if x.upper() == 'SHRINK']
    if not i: sys.exit('no SHRINK found in the template input')
    i = i[-1]; s = l[i+1]; block = 'NEWK\n' + s + '\n'
    if s.split()[0] == '0': block += l[i+2] + '\n'
    return block + '0 0\n'

def write_gui(fname, lat, conv, pos):
    s = '3 1 1\n' + ''.join('%20.12f%20.12f%20.12f\n' % tuple(v) for v in lat)
    s += '1\n  1.00  0.00  0.00\n  0.00  1.00  0.00\n  0.00  0.00  1.00\n  0.00  0.00  0.00\n'
    s += '%d\n' % len(conv) + ''.join('%5d%20.12f%20.12f%20.12f\n' % (c, *p) for c, p in zip(conv, pos))
    open(fname, 'w').write(s)

def write_d12(fname, title, template):
    open(fname, 'w').write(f'{title}\nEXTERNAL\nEND\n{template}\nGRADCAL\nEND\n')

# =============================================================================
#  SITE SETTINGS - adapt to your cluster (or set the environment variables)
# =============================================================================
CRYSTAL_ENV = os.environ.get("CRYFREQ_CRYSTAL_ENV",
    "~/sw/crystal/crystal23_v1_0_1_Linux-ifort21.4_openmpi4.1.1_exe/utils23/cry23.bashrc")
RUN_PCRYSTAL = os.environ.get("CRYFREQ_RUN_PCRYSTAL",
    "~/sw/crystal/crystal23_v1_0_1_Linux-ifort21.4_openmpi4.1.1_exe/utils23/runPcry23")
PARTITION = os.environ.get("CRYFREQ_PARTITION", "mpi,normal")
MEM_PER_CPU = os.environ.get("CRYFREQ_MEM_PER_CPU", "6400M")

JOB = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={cores}
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time}
#SBATCH --array=0-{n}%{par}          # 0 = undisplaced reference, 1..{n} = displacements

source {crystal_env}
if [ ${{SLURM_ARRAY_TASK_ID}} -eq 0 ]; then d=ref; else d=$(printf "disp-%03d" ${{SLURM_ARRAY_TASK_ID}}); fi
cd ${{SLURM_SUBMIT_DIR}}/${{d}}
{run_pcrystal} {cores} struc
{polar}"""

POLAR_STEP = """
# Berry phase, preliminary run (POLARI): serial properties on the wave function of this structure
PROP=${CRY23_EXEDIR}/${VERSION}/properties
rm -rf polrun && mkdir polrun && cd polrun
cp ../struc.f9 fort.9
$PROP < ../polari.d3 > ../polari.outp 2>&1
cp fort.27 ../pol.f27
cd .. && rm -rf polrun
"""

SPOLBP_JOB = """#!/bin/bash
#SBATCH --job-name={name}_spolbp
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu={mem}
#SBATCH --time={time}
# Berry phase, third run (SPOLBP): polarization difference displaced - reference, for every displacement.
# Submit after run_array.sh has finished completely.
source {crystal_env}
PROP=${{CRY23_EXEDIR}}/${{VERSION}}/properties
cd ${{SLURM_SUBMIT_DIR:-.}}
for d in disp-*; do
  (cd $d && rm -rf sprun && mkdir sprun && cd sprun &&
   cp ../struc.f9 fort.9 && cp ../../ref/pol.f27 fort.28 && cp ../pol.f27 fort.29 &&
   $PROP < ../spolbp.d3 > ../spolbp.outp 2>&1; cd .. && rm -rf sprun)
  echo "$d done"
done
"""

# ---------------------------------------------------------------- setup
def cmd_setup(a):
    from phonopy import Phonopy
    lat, conv, pos = read_crystal_geometry(a.geom)
    template = split_input(a.input)
    cell = make_cell(lat, conv, pos)
    ph = Phonopy(cell, supercell_matrix=np.eye(3, dtype=int), primitive_matrix=None)
    ph.generate_displacements(distance=a.dist, is_plusminus=True if a.pm else 'auto')
    sg = ph.symmetry.get_international_table()
    cells = ph.supercells_with_displacements
    os.makedirs('ref', exist_ok=True)
    write_gui('ref/struc.gui', lat, conv, pos)
    write_d12('ref/struc.d12', 'cryfreq reference (undisplaced)', template)
    for i, c in enumerate(cells, start=1):
        d = f'disp-{i:03d}'; os.makedirs(d, exist_ok=True)
        write_gui(f'{d}/struc.gui', c.cell, conv, c.positions)
        write_d12(f'{d}/struc.d12', f'cryfreq displacement {i}', template)
    ph.save('phonopy_disp.yaml')
    np.savetxt('conv_numbers.dat', conv, fmt='%d')
    if a.polar:
        nk = newk_from_template(template)
        for d in ['ref'] + [f'disp-{i:03d}' for i in range(1, len(cells)+1)]:
            open(f'{d}/polari.d3', 'w').write(nk + 'POLARI\nEND\n')
            if d != 'ref': open(f'{d}/spolbp.d3', 'w').write('SPOLBP\nEND\n')
        open('run_spolbp.sh', 'w').write(SPOLBP_JOB.format(name=a.name, time='01:00:00', partition=PARTITION, mem=MEM_PER_CPU, crystal_env=CRYSTAL_ENV))
    open('run_array.sh', 'w').write(JOB.format(name=a.name, cores=a.cores, time=a.time, n=len(cells), par=a.par,
                                               partition=PARTITION, mem=MEM_PER_CPU, crystal_env=CRYSTAL_ENV, run_pcrystal=RUN_PCRYSTAL,
                                               polar=POLAR_STEP if a.polar else ''))
    print(f'{len(conv)} atoms, space group {sg}: {len(cells)} displacements of {a.dist} A '
          f'({"plus/minus" if a.pm else "phonopy auto"}) + undisplaced reference')
    print('submit with:  sbatch run_array.sh' + ('   and afterwards:  sbatch run_spolbp.sh' if a.polar else ''))

# ---------------------------------------------------------------- status
def cmd_status(a):
    import phonopy
    ph = phonopy.load('phonopy_disp.yaml', produce_fc=False, log_level=0)
    nat = len(ph.supercell.numbers); n = len(ph.supercells_with_displacements)
    done = 0
    for d in ['ref'] + [f'disp-{i:03d}' for i in range(1, n+1)]:
        f = read_forces(f'{d}/struc.out', nat) if os.path.exists(f'{d}/struc.out') else None
        ok = f is not None; done += ok and d != 'ref'
        pol = 'POLARI ok' if os.path.exists(f'{d}/pol.f27') else 'POLARI -'
        sp = '' if d == 'ref' else ('  SPOLBP ok' if os.path.exists(f'{d}/spolbp.outp') else '  SPOLBP -')
        print(f'{d:9s} {"forces ok" if ok else "forces MISSING"}  {pol}{sp}' + (f'   max|F| = {np.abs(f).max():.2e} eV/A' if ok and d == 'ref' else ''))
    print(f'{done}/{n} displacements finished')

# ---------------------------------------------------------------- Born charges
def cart_symops(ph):
    ds = ph.symmetry.dataset
    rots = ds['rotations'] if isinstance(ds, dict) else ds.rotations      # older/newer spglib
    trs = ds['translations'] if isinstance(ds, dict) else ds.translations
    L = ph.supercell.cell                                   # rows = lattice vectors
    R = [L.T @ r @ np.linalg.inv(L.T) for r in rots]
    return R, rots, trs

def atom_map(ph, rot, tr):
    sp = ph.supercell.scaled_positions; n = len(sp); m = np.zeros(n, int)
    for k in range(n):
        new = (rot @ sp[k] + tr)
        d = (sp - new + 0.5) % 1 - 0.5
        m[k] = np.argmin(np.linalg.norm(d @ ph.supercell.cell, axis=1))
    return m

def born_from_dipoles(ph, dipoles):
    """dipoles[i] = change of the cell dipole (e*Angstrom) for displacement i+1 relative to ref."""
    disp = ph.dataset['first_atoms']; nat = len(ph.supercell.numbers)
    Rc, Rf, T = cart_symops(ph)
    eqs = [[] for _ in range(nat)]
    for i, d in enumerate(disp):
        k = d['number']; u = np.array(d['displacement']); D = np.array(dipoles[i])
        for rc, rf, t in zip(Rc, Rf, T):
            m = atom_map(ph, rf, t)
            eqs[m[k]].append((rc @ u, rc @ D))
    Z = np.zeros((nat, 3, 3))
    for k in range(nat):
        U = np.array([e[0] for e in eqs[k]]).T; D = np.array([e[1] for e in eqs[k]]).T
        if np.linalg.matrix_rank(U, 1e-6) < 3: sys.exit(f'atom {k+1}: displacements do not span 3D')
        Z[k] = D @ np.linalg.pinv(U)
    asr = Z.sum(axis=0)                                       # should be zero (charge neutrality)
    print(f'Born charges: acoustic sum rule before correction, max |sum_k Z_k| = {np.abs(asr).max():.3f} e '
          f'(per atom {np.abs(asr).max()/nat:.3f} e)')
    Z -= Z.mean(axis=0)                                       # acoustic sum rule
    return Z

AU_POL_TO_CM2 = 57.21476                # e/bohr^2 -> C/m^2
def read_spolbp(fname):
    """Polarization difference P(displaced) - P(reference) in C/m^2 from a SPOLBP output.
    Format (CRYSTAL23):  'POLARIZATION, P, PROJECTED ALONG THE CARTESIAN AXES (C/m**2)'
                         '   X  Y  Z' / '   P   px  py  pz' / '   QUANTUM  qx qy qz'"""
    t = open(fname).read().split('\n')
    hdr = [i for i, l in enumerate(t) if 'POLARIZATION, P, PROJECTED ALONG THE CARTESIAN AXES' in l.upper()]
    if not hdr: sys.exit(f'{fname}: no SPOLBP polarization block found')
    if 'C/M**2' not in t[hdr[-1]].upper(): sys.exit(f'{fname}: unexpected units in "{t[hdr[-1]].strip()}"')
    for l in t[hdr[-1]+1: hdr[-1]+6]:
        p = l.split()
        if len(p) == 4 and p[0] == 'P':
            return np.array(list(map(float, p[1:4])))
    sys.exit(f'{fname}: line "P px py pz" not found after the polarization header')

def read_born_file(fname, nat):
    """Plain file: one line per atom with 9 numbers (Zxx Zxy Zxz Zyx ... ), rows = field/dipole direction."""
    z = np.loadtxt(fname)
    if z.shape != (nat, 9): sys.exit(f'{fname}: expected {nat} lines with 9 numbers')
    return z.reshape(nat, 3, 3)

# ---------------------------------------------------------------- analyze
def load_phonopy_with_fc(no_ref=False, quiet=False):
    """phonopy object with force constants from the disp-XXX outputs (+ reference correction)."""
    import phonopy
    ph = phonopy.load('phonopy_disp.yaml', produce_fc=False, log_level=0)
    nat = len(ph.supercell.numbers); n = len(ph.supercells_with_displacements)
    F = []
    for i in range(1, n+1):
        f = read_forces(f'disp-{i:03d}/struc.out', nat)
        if f is None: sys.exit(f'disp-{i:03d}: no forces - run "cryfreq.py status"')
        F.append(f)
    F = np.array(F)
    if os.path.exists('ref/struc.out') and not no_ref:
        f0 = read_forces('ref/struc.out', nat)
        if f0 is not None:
            F -= f0
            if not quiet: print(f'residual forces of reference subtracted (max {np.abs(f0).max():.2e} eV/A)')
    F -= F.mean(axis=1, keepdims=True)                        # remove drift
    ph.forces = F
    ph.produce_force_constants(); ph.symmetrize_force_constants()
    return ph, nat, n

def cmd_analyze(a):
    ph, nat, n = load_phonopy_with_fc(a.no_ref)
    ph.run_qpoints([[0, 0, 0]], with_eigenvectors=True)
    freqs = ph.qpoints.frequencies[0]; evec = ph.qpoints.eigenvectors[0]
    w = freqs * THZ_TO_CM                                      # cm-1
    # irreducible representations
    labels = ['?'] * len(w); degsets = None
    try:
        (ph.run_irreps if hasattr(ph, "run_irreps") else ph.set_irreps)([0, 0, 0], degeneracy_tolerance=1e-3)
        ir = ph.irreps
        degsets = [list(d) for d in ir._degenerate_sets]
        for deg, lab in zip(ir._degenerate_sets, ir._ir_labels):
            for j in deg: labels[j] = lab if lab else '?'
    except Exception as e:
        print('irreps not available:', e)
    # Born charges / intensities
    Z = None
    if a.born: Z = read_born_file(a.born, nat)
    elif a.dipoles: Z = born_from_dipoles(ph, np.loadtxt(a.dipoles).reshape(n, 3))
    elif a.polar:
        vol = abs(np.linalg.det(ph.supercell.cell))
        dP = np.array([read_spolbp(f'disp-{i:03d}/spolbp.outp') for i in range(1, n+1)])
        D = dP * vol / 16.02176634                     # C/m^2 * A^3  ->  e*A
        np.savetxt('dipoles_from_spolbp.dat', D, fmt='%14.8f', header='cell dipole change (e*A) per displacement')
        Z = born_from_dipoles(ph, D)
    I = np.full(len(w), np.nan)
    if Z is not None:
        m = ph.supercell.masses
        E = evec.T.reshape(len(w), nat, 3)                    # mode, atom, xyz
        p = np.einsum('kab,nkb->na', Z, E / np.sqrt(m)[None, :, None])
        I = E2_AMU_TO_KMMOL * (np.abs(p) ** 2).sum(axis=1)
        np.savetxt('born_charges.dat', Z.reshape(nat, 9), fmt='%10.5f',
                   header='Born effective charges (e), one line per atom: xx xy xz yx yy yz zx zy zz')
    # group degenerate modes: symmetry-based sets from phonopy if available, otherwise frequency tolerance
    groups = []
    if degsets:
        groups = [(min(s), max(s)) for s in degsets]
    else:
        j = 0
        while j < len(w):
            k = j
            while k + 1 < len(w) and abs(w[k+1] - w[j]) < a.degtol: k += 1
            groups.append((j, k)); j = k + 1
    # mode character: share of each mode on every set of symmetry-equivalent atoms (|e|^2, sums to 100 %)
    try:
        rep = ph.symmetry.get_map_atoms()
        sets = sorted(set(rep)); syms = ph.supercell.symbols
        names = [f'{syms[r]}{r+1}' for r in sets]
        E2 = np.abs(evec.T.reshape(len(w), nat, 3)) ** 2
        share = np.array([[E2[n, rep == r].sum() for r in sets] for n in range(len(w))]) * 100
        with open('mode_character.dat', 'w') as f:
            f.write('# share (%) of each mode on the sets of symmetry-equivalent atoms (label = first atom of the set)\n')
            f.write('#  modes   freq   irrep  I(km/mol) ' + ''.join(f'{x:>8s}' for x in names) + '\n')
            for j, k in groups:
                Ii = np.nansum(I[j:k+1]) if Z is not None else float('nan')
                f.write(f'{j+1:4d}-{k+1:<4d}{w[j:k+1].mean():8.1f} {labels[j]:>5s} {Ii:9.1f} ' + ''.join(f'{x:8.1f}' for x in share[j:k+1].mean(axis=0)) + '\n')
    except Exception as e:
        print('mode character skipped:', e)
    with open('frequencies.dat', 'w') as f:
        f.write('# modes   freq(cm-1)  irrep   IR_intensity(km/mol, summed over degenerate modes)\n')
        for j, k in groups:
            f.write(f'{j+1:4d}-{k+1:<4d} {w[j:k+1].mean():10.2f}  {labels[j]:>6s}  {np.nansum(I[j:k+1]) if Z is not None else float("nan"):12.2f}\n')
    print(open('frequencies.dat').read())
    if (w[3:] < -1).any(): print('WARNING: imaginary frequencies beyond the acoustic modes!')
    if Z is not None:
        x = np.arange(a.wmin, a.wmax, 0.5); g = a.fwhm / 2
        S = sum(I[i] * g / np.pi / ((x - w[i]) ** 2 + g ** 2) for i in range(3, len(w)))
        np.savetxt('ir_spectrum.dat', np.c_[x, S], fmt='%10.3f', header=f'wavenumber(cm-1)  absorbance (Lorentzian, FWHM {a.fwhm} cm-1)')
        try:
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(x, S, 'k', lw=1)
            ax.vlines(w[3:], 0, I[3:] / I[3:].max() * S.max() * 0.3, color='r', lw=0.8)
            ax.set_xlabel('wavenumber (cm$^{-1}$)'); ax.set_ylabel('IR absorbance (arb. u.)'); ax.set_xlim(a.wmax, a.wmin)
            fig.savefig('ir_spectrum.png', dpi=200, bbox_inches='tight')
        except Exception as e: print('plot skipped:', e)
    if a.compare:
        rows = read_freqcalc(a.compare)
        if not rows: print('no FREQCALC mode table found in', a.compare); return
        mine = [dict(j=j, k=k, w=w[j:k+1].mean(), lab=labels[j], n=k-j+1,
                     I=np.nansum(I[j:k+1]) if Z is not None else float('nan')) for j, k in groups]
        used = set()
        print('\ncomparison with FREQCALC (', a.compare, ') - modes matched by irrep and frequency')
        print('  irrep   FREQCALC   cryfreq    diff  |  I_CRYSTAL  I_cryfreq   (km/mol)')
        dev = []; dI = []
        for r in rows:
            if abs(r['freq']) < 1.0: continue                          # acoustic modes
            cand = [m for m in mine if m['j'] not in used and abs(m['w']) >= 1.0]
            same = [m for m in cand if m['lab'].upper() == r['irrep'].upper()]
            pool = same if same else cand
            m = min(pool, key=lambda x: abs(x['w'] - r['freq'])); used.add(m['j'])
            n = r['last'] - r['first'] + 1
            dev.append(m['w'] - r['freq'])
            flag = '' if same else '   (no label match)'
            print(f'  {r["irrep"]:>5s} {r["freq"]:10.2f} {m["w"]:10.2f} {m["w"] - r["freq"]:7.2f}  | {r["intens"]:10.2f} {m["I"]:10.2f}{flag}')
            if Z is not None and r['intens'] > 1.0: dI.append(m['I'] / r['intens'] - 1)
        dev = np.array(dev)
        print(f'frequencies: max |deviation| {np.abs(dev).max():.2f} cm-1, mean {np.abs(dev).mean():.2f} cm-1')
        if dI: print(f'IR intensities of active modes: max |relative deviation| {100*np.abs(dI).max():.1f} %, mean {100*np.abs(dI).mean():.1f} %')


# ---------------------------------------------------------------- thermodynamics
KB_EV = 8.617333262e-5; CM_TO_EV = 1.239841984e-4; EV_TO_KJMOL = 96.48533212; HARTREE_EV = 27.211386245988

def read_scf_energy(fname):
    """last SCF total energy (Hartree) of a CRYSTAL output"""
    E = None
    for l in open(fname):
        if 'TOTAL ENERGY(DFT)(AU)' in l:
            m = re.search(r'\)\s*(-?\d+\.\d+E[+-]\d+)', l)
            if m: E = float(m.group(1))
    return E

def harmonic_gamma(w_cm, T):
    """ZPE, U, F, S, Cv from Gamma-point frequencies (cm-1); acoustic and imaginary modes skipped.
    Returns eV per cell (S, Cv in eV/K per cell)."""
    w = np.asarray(w_cm); w = w[w > 1.0]
    e = w * CM_TO_EV; zpe = 0.5 * e.sum()
    T = np.atleast_1d(np.asarray(T, float)); res = []
    for t in T:
        if t < 1e-6: res.append((zpe, zpe, zpe, 0.0, 0.0)); continue
        x = e / (KB_EV * t); ex = np.exp(-x)
        F = zpe + KB_EV * t * np.log1p(-ex).sum()
        U = zpe + (e * ex / (1 - ex)).sum()
        S = KB_EV * (x * ex / (1 - ex) - np.log1p(-ex)).sum()
        Cv = KB_EV * (x**2 * ex / (1 - ex)**2).sum()
        res.append((zpe, U, F, S, Cv))
    return np.array(res)

def cmd_thermo(a):
    ph, nat, n = load_phonopy_with_fc(a.no_ref)
    T = np.arange(a.tmin, a.tmax + 0.5 * a.tstep, a.tstep)
    E_el = read_scf_energy('ref/struc.out') if os.path.exists('ref/struc.out') else None
    E_el_eV = E_el * HARTREE_EV if E_el is not None else np.nan
    if a.mesh:
        ph.run_mesh(a.mesh, is_gamma_center=True)
        ph.run_thermal_properties(t_min=a.tmin, t_max=a.tmax, t_step=a.tstep)
        tp = ph.get_thermal_properties_dict()
        Tm = tp['temperatures']
        F = tp['free_energy'] / EV_TO_KJMOL; S = tp['entropy'] / EV_TO_KJMOL / 1000; Cv = tp['heat_capacity'] / EV_TO_KJMOL / 1000
        U = F + Tm * S; zpe = F[0] if abs(Tm[0]) < 1e-6 else np.nan
        method = f'phonon DOS on a {a.mesh[0]}x{a.mesh[1]}x{a.mesh[2]} q-mesh (Fourier-interpolated force constants of this cell)'
        T = Tm
    else:
        ph.run_qpoints([[0, 0, 0]]); w = ph.qpoints.frequencies[0] * THZ_TO_CM
        if (w[3:] < -1).any(): print('WARNING: imaginary optical modes are skipped - the structure is not a minimum!')
        r = harmonic_gamma(w, T); zpe, U, F, S, Cv = r[0, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 4]
        method = 'Gamma-point frequencies of this cell (acoustic modes excluded)'
    with open('thermo.dat', 'w') as f:
        f.write(f'# harmonic vibrational thermodynamics, {method}\n')
        f.write(f'# per cell ({nat} atoms); E_el = {E_el} Ha from ref/struc.out\n')
        f.write('#  T/K     F_vib/eV    U_vib/eV    S_vib/(meV/K)  Cv/(meV/K)   F_vib/(kJ/mol)   E_el+F_vib/eV\n')
        for i in range(len(T)):
            f.write(f'{T[i]:7.1f} {F[i]:11.5f} {U[i]:11.5f} {1000*S[i]:13.5f} {1000*Cv[i]:11.5f} {F[i]*EV_TO_KJMOL:15.3f} {E_el_eV + F[i]:17.6f}\n')
    print(f'method: {method}')
    print(f'zero-point vibrational energy: {zpe:.5f} eV per cell = {zpe*EV_TO_KJMOL:.2f} kJ/mol (per cell)')
    for t in a.at:
        i = np.argmin(abs(T - t))
        print(f'T = {T[i]:6.1f} K:  F_vib = {F[i]:.5f} eV,  T*S_vib = {T[i]*S[i]:.5f} eV,  E_el + F_vib = {E_el_eV + F[i]:.6f} eV')
    print('written: thermo.dat  (G = F at this level: harmonic, fixed volume, pV neglected)')

def cmd_compare(a):
    """difference of two thermo.dat files (A - B), e.g. Fe_oct - Fe_tet"""
    A = np.loadtxt(os.path.join(a.dirA, 'thermo.dat')); B = np.loadtxt(os.path.join(a.dirB, 'thermo.dat'))
    if A.shape != B.shape or np.abs(A[:, 0] - B[:, 0]).max() > 1e-6: sys.exit('temperature grids differ - rerun thermo with identical settings')
    print(f'# differences {a.dirA} - {a.dirB}  (meV per cell)')
    print('#  T/K    dE_el     dF_vib    dE_el+dF_vib    -T*dS_vib')
    dEel = 1000 * ((A[0, 6] - A[0, 1]) - (B[0, 6] - B[0, 1]))
    for ra, rb in zip(A, B):
        if a.at and not any(abs(ra[0] - t) < 1e-6 for t in a.at): continue
        dF = 1000 * (ra[1] - rb[1]); dTS = -ra[0] * (ra[3] - rb[3])
        print(f'{ra[0]:7.1f} {dEel:9.2f} {dF:9.2f} {dEel + dF:14.2f} {dTS:12.2f}')

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    s = p.add_subparsers(dest='cmd', required=True)
    q = s.add_parser('setup'); q.add_argument('--geom', required=True, help='CRYSTAL output of the optimization (final geometry is used)')
    q.add_argument('--input', required=True, help='CRYSTAL input whose basis set and Hamiltonian (everything after the geometry block) is used')
    q.add_argument('--dist', type=float, default=0.01, help='displacement in Angstrom (default 0.01)')
    q.add_argument('--no-pm', dest='pm', action='store_false', help='one-sided displacements where symmetry allows')
    q.add_argument('--cores', type=int, default=64); q.add_argument('--time', default='06:00:00')
    q.add_argument('--par', type=int, default=6, help='max. simultaneous array tasks'); q.add_argument('--name', default='cryfreq')
    q.add_argument('--no-polar', dest='polar', action='store_false', help='skip the Berry-phase (POLARI/SPOLBP) steps')
    s.add_parser('status')
    q = s.add_parser('analyze')
    q.add_argument('--born', help='Born charges, plain file (natoms x 9)')
    q.add_argument('--dipoles', help='cell-dipole changes (e*A) per displacement, one line "dx dy dz" each')
    q.add_argument('--polar', action='store_true', help='Born charges from the SPOLBP outputs (disp-XXX/spolbp.outp)')
    q.add_argument('--compare', help='CRYSTAL FREQCALC output for comparison')
    q.add_argument('--no-ref', action='store_true', help='do not subtract residual forces of ref/')
    q.add_argument('--fwhm', type=float, default=10.0); q.add_argument('--wmin', type=float, default=100); q.add_argument('--wmax', type=float, default=1200)
    q.add_argument('--degtol', type=float, default=0.5, help='cm-1 tolerance for grouping degenerate modes')
    q = s.add_parser('thermo', help='ZPE and harmonic vibrational free energy')
    q.add_argument('--mesh', type=int, nargs=3, help='q-mesh for the phonon DOS (default: Gamma point only)')
    q.add_argument('--tmin', type=float, default=0); q.add_argument('--tmax', type=float, default=1200); q.add_argument('--tstep', type=float, default=50)
    q.add_argument('--at', type=float, nargs='*', default=[300, 1100], help='temperatures to print (must lie on the T grid)')
    q.add_argument('--no-ref', action='store_true')
    q = s.add_parser('compare', help='differences A - B of two thermo.dat (same settings!)')
    q.add_argument('dirA'); q.add_argument('dirB'); q.add_argument('--at', type=float, nargs='*', default=[])
    a = p.parse_args()
    {'setup': cmd_setup, 'status': cmd_status, 'analyze': cmd_analyze, 'thermo': cmd_thermo, 'compare': cmd_compare}[a.cmd](a)

if __name__ == '__main__':
    main()
