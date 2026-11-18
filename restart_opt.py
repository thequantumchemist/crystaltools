#!/usr/bin/env python3
"""
restart_opt.py - restart an interrupted CRYSTAL geometry optimization from its output.

Reads the original input (struc.d12) and the output of the interrupted run (struc.out),
takes the optimization point with the lowest completed energy (or the one given with --point),
and writes
   struc.gui  - that geometry in EXTERNAL (fort.34) format, all atoms, with the symmetry
                operators of the run, so the cell keeps its symmetry and atom numbering
   struc.d12  - the same input, but with EXTERNAL instead of the old geometry block
                (CRYSTAL/SUPERCEL/ATOMSUBS ...); the OPTGEOM block, basis sets and
                Hamiltonian are copied unchanged
The old files are kept as struc_run1.d12 / struc_run1.out (run2, run3 ... if those exist).

usage:  python restart_opt.py [--d12 struc.d12] [--out struc.out] [--point N]
"""
import argparse, os, re, sys
import numpy as np

OPEN_BLOCKS = {'OPTGEOM', 'FREQCALC', 'ANHARM', 'ELASTCON', 'EOS', 'IRSPEC', 'RAMSPEC', 'PREOPTGEOM'}

def lattice(a, b, c, al, be, ga):
    al, be, ga = np.radians([al, be, ga])
    cx = c*np.cos(be); cy = c*(np.cos(al) - np.cos(be)*np.cos(ga))/np.sin(ga)
    return np.array([[a, 0, 0], [b*np.cos(ga), b*np.sin(ga), 0], [cx, cy, np.sqrt(c*c - cx*cx - cy*cy)]])

def parse_output(fname):
    t = open(fname).read().split('\n')
    # symmetry operators (fractional, crystallographic frame) - first block
    i = next((k for k, l in enumerate(t) if 'SYMMOPS - TRANSLATORS IN FRACTIONAL UNITS' in l), None)
    if i is None: sys.exit('no symmetry operators found in ' + fname)
    nops = int(t[i].split()[1]); ops = []
    for l in t[i+3:i+3+nops]:
        v = list(map(float, l.split()[2:]))
        ops.append((np.array(v[:9]).reshape(3, 3), np.array(v[9:12])))
    # geometry blocks: lattice parameters + all atoms, followed (if completed) by the energy
    pts = []
    for k, l in enumerate(t):
        if 'LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR' in l and k+3 < len(t):
            cell = list(map(float, t[k+3].split()))
            j = k + 4
            while j < len(t) and 'ATOMS IN THE ASYMMETRIC UNIT' not in t[j]: j += 1
            if j >= len(t): break
            nat = int(t[j].split(':')[-1]); atoms = []; j += 3
            for l2 in t[j:j+nat]:
                p = l2.split()
                if len(p) < 7: break
                atoms.append((int(p[2]), p[3], np.array(list(map(float, p[4:7])))))
            if len(atoms) != nat: continue
            # energy of this point: next TOTAL ENERGY(DFT) before the next geometry block
            E = None; m = j + nat
            while m < len(t) and 'LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR' not in t[m]:
                if 'TOTAL ENERGY(DFT)(AU)' in t[m]:
                    E = float(re.search(r'\)\s*(-?\d+\.\d+E[+-]\d+)', t[m]).group(1))
                if 'POINT' in t[m] and 'OPTIMIZATION' in t[m] and m > j + nat: break
                m += 1
            label = 'final' if any('FINAL OPTIMIZED GEOMETRY' in x for x in t[max(0, k-4):k]) else ''
            pts.append(dict(line=k, cell=cell, atoms=atoms, E=E, label=label))
    # drop the duplicate printouts that are not optimization points (keep last print of each geometry)
    return ops, pts

def write_gui(fname, cell, atoms, ops):
    L = lattice(*cell); L[np.abs(L) < 1e-10] = 0.0
    ang = np.array(cell[3:])
    ctype = 1 if len(ops) == 1 else (3 if np.abs(ang - 90).max() < 1.0 else 2)
    s = f'   3   1   {ctype}\n' + ''.join(f'{v[0]:20.12E}{v[1]:20.12E}{v[2]:20.12E}\n' for v in L)
    s += f'{len(ops):5d}\n'
    for R, tr in ops:
        Rc = L.T @ R @ np.linalg.inv(L.T); tc = tr @ L
        Rc[np.abs(Rc) < 1e-10] = 0.0; tc[np.abs(tc) < 1e-10] = 0.0
        s += ''.join(f'{r[0]:20.12E}{r[1]:20.12E}{r[2]:20.12E}\n' for r in Rc)
        s += f'{tc[0]:20.12E}{tc[1]:20.12E}{tc[2]:20.12E}\n'
    s += f'{len(atoms):5d}\n'
    for z, _, f in atoms:
        x = f @ L; s += f'{z:5d}{x[0]:20.12E}{x[1]:20.12E}{x[2]:20.12E}\n'
    open(fname, 'w').write(s)

def new_input(d12):
    lines = open(d12).read().split('\n')
    depth = 0; gend = None; opt = []; inopt = False
    for i, l in enumerate(lines[1:], start=1):
        w = l.strip().upper()
        if w in OPEN_BLOCKS:
            depth += 1
            if w == 'OPTGEOM': inopt = True
        elif w.startswith('END'):
            if depth > 0:
                depth -= 1
                if inopt and depth == 0: opt.append(l); inopt = False; continue
            else: gend = i; break
        if inopt: opt.append(l)
    if gend is None: sys.exit('end of the geometry block not found in ' + d12)
    if not opt: print('WARNING: no OPTGEOM block found - the new input will only do a single point')
    title = lines[0].rstrip() + ' (restart)'
    return '\n'.join([title, 'EXTERNAL'] + opt + ['END'] + lines[gend+1:])

def backup(name):
    if not os.path.exists(name): return None
    base, ext = os.path.splitext(name); n = 1
    while os.path.exists(f'{base}_run{n}{ext}'): n += 1
    os.rename(name, f'{base}_run{n}{ext}'); return f'{base}_run{n}{ext}'

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--d12', default='struc.d12'); p.add_argument('--out', default='struc.out')
    p.add_argument('--point', type=int, help='use this point (1 = first geometry) instead of the lowest energy')
    a = p.parse_args()
    ops, pts = parse_output(a.out)
    if not pts: sys.exit('no geometry found in ' + a.out)
    print(f'{len(ops)} symmetry operators, {len(pts[0]["atoms"])} atoms, {len(pts)} geometries in {a.out}')
    for n, q in enumerate(pts, 1):
        E = f'{q["E"]:.10f}' if q['E'] is not None else 'not completed'
        print(f'  point {n:3d}   E = {E}   V = {np.linalg.det(lattice(*q["cell"])):.3f} A^3 {q["label"]}')
    if a.point: sel = pts[a.point - 1]
    else:
        done = [q for q in pts if q['E'] is not None]
        sel = min(done, key=lambda q: q['E']) if done else pts[-1]
    n = pts.index(sel) + 1
    print(f'-> using point {n}' + (f' (E = {sel["E"]:.10f})' if sel['E'] is not None else ''))
    if sel['atoms'][0][1] != pts[0]['atoms'][0][1]: print('WARNING: atom order differs from the first geometry')
    txt = new_input(a.d12)
    b1 = backup(a.d12); b2 = backup(a.out)
    gui = os.path.splitext(a.d12)[0] + '.gui'
    if os.path.exists(gui): backup(gui)
    write_gui(gui, sel['cell'], sel['atoms'], ops)
    open(a.d12, 'w').write(txt + ('' if txt.endswith('\n') else '\n'))
    print(f'written: {a.d12} (EXTERNAL + OPTGEOM), {gui};  old files kept as {b1}, {b2}')
    print('atom 1 is', sel['atoms'][0][1], '- ATOMSPIN/projection labels stay valid if this is as before')

if __name__ == '__main__':
    main()
