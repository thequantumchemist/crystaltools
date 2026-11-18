import numpy as np, itertools, sys, yaml
sys.path.insert(0,'../../..'); import cryfreq as cf
lat,num,cart0=cf.read_crystal_geometry('fake_opt.out')
nat=len(num); m=np.array([cf.MASSES['Al' if n==13 else 'O'] for n in num])
bonds=[]
for i in range(nat):
    for j in range(nat):
        for s in itertools.product([-1,0,1],repeat=3):
            if i==j and s==(0,0,0): continue
            v=cart0[j]+np.array(s)@lat-cart0[i]; r=np.linalg.norm(v)
            if r<2.05: k=6.0
            elif r<2.9: k=1.5 if (num[i]==8 and num[j]==8) else 0.8
            else: continue
            bonds.append((i,j,np.array(s),k,r))
def forces(cart):   # E = sum over ordered pairs 0.25 k (r-r0)^2  (each unordered pair counted twice)
    F=np.zeros((nat,3))
    for i,j,s,k,r0 in bonds:
        v=cart[j]+s@lat-cart[i]; r=np.linalg.norm(v)
        f=0.5*k*(r-r0)*v/r      # force on i (dE/d r_i = -0.5 k dr v/r)
        F[i]+=f; F[j]-=f
    return F
h=1e-4; H=np.zeros((3*nat,3*nat))
for a in range(nat):
    for b in range(3):
        cp=cart0.copy(); cp[a,b]+=h; cm=cart0.copy(); cm[a,b]-=h
        H[:,3*a+b]=-(forces(cp)-forces(cm)).ravel()/(2*h)
H=0.5*(H+H.T); Mw=1/np.sqrt(np.repeat(m,3)); D=H*np.outer(Mw,Mw)
ev,evecs=np.linalg.eigh(D)
conv=np.sqrt(1.602176634e-19/1e-20/1.66053906660e-27)/(2*np.pi*2.99792458e10)
wref=np.sign(ev)*np.sqrt(np.abs(ev))*conv
Zm={13:3.0,8:-2.0}
d=yaml.safe_load(open('phonopy_disp.yaml'))['displacements']
def fake_out(fname,F):
    s=' CARTESIAN FORCES IN HARTREE/BOHR (ANALYTICAL)\n   ATOM                     X                   Y                   Z\n'
    s+=''.join('%4d%4d%20.12E%20.12E%20.12E\n'%(i+1,num[i],*(f/cf.HARTREE_BOHR_TO_EV_ANG)) for i,f in enumerate(F))
    open(fname,'w').write(s)
fake_out('ref/struc.out',forces(cart0)); dip=[]
for i,x in enumerate(d,start=1):
    c=np.array([[float(v) for v in l.split()[1:4]] for l in open(f'disp-{i:03d}/struc.gui').read().split('\n')[10:10+nat]])
    fake_out(f'disp-{i:03d}/struc.out',forces(c))
    dip.append(Zm[num[x['atom']-1]]*np.array(x['displacement']))
np.savetxt('dipoles.dat',np.array(dip))
Z=np.array([np.eye(3)*Zm[n] for n in num]); I=[]
for n in range(3*nat):
    e=evecs[:,n].reshape(nat,3)/np.sqrt(m)[:,None]
    I.append(cf.E2_AMU_TO_KMMOL*(np.einsum('kab,kb->a',Z,e)**2).sum())
I=np.array(I); lines=[]; j=0
while j<len(wref):
    k=j
    while k+1<len(wref) and abs(wref[k+1]-wref[j])<0.5: k+=1
    lines.append('%5d-%4d    %12.4E   %10.4f %10.4f  (%-3s)   %s (%9.2f)   I\n'%(j+1,k+1,0.0,wref[j:k+1].mean(),wref[j]/33.356,'X','A' if I[j:k+1].sum()>1e-3 else 'I',I[j:k+1].sum()))
    j=k+1
open('fake_freq.out','w').write('    MODES         EIGV          FREQUENCIES     IRREP  IR   INTENS    RAMAN\n             (HARTREE**2)   (CM**-1)     (THZ)             (KM/MOL)\n'+''.join(lines)+'\n')
