
import os, os.path as osp
import numpy as np
import uptools



class Bunch:
    def __init__(self, **kwargs):
        self.arrays = kwargs

    def __getattr__(self, name):
       return self.arrays[name]

    def __getitem__(self, where):
        """Selection mechanism"""
        new = self.__class__()
        new.arrays = {k: v[where] for k, v in self.arrays.items()}
        return new

    def __len__(self):
        for k, v in self.arrays.items():
            try:
                return len(v)
            except TypeError:
                return 1


class FourVectorArray:
    """
    Wrapper class for Bunch, with more specific 4-vector stuff
    """
    def __init__(self, pt, eta, phi, energy, **kwargs):
        self.bunch = Bunch(
            pt=pt, eta=eta, phi=phi, energy=energy, **kwargs
            )

    def __getattr__(self, name):
       return getattr(self.bunch, name)

    def __getitem__(self, where):
        new = self.__class__([], [], [], [])
        new.bunch = self.bunch[where]
        return new

    def __len__(self):
        return len(self.bunch)


def is_array(a):
    """
    Checks if a thing is an array or maybe a number
    """
    try:
        shape = a.shape
        return len(shape) >= 1
    except AttributeError:
        return False


def calc_dphi(phi1, phi2):
    """
    Calculates delta phi. Assures output is within -pi .. pi.
    """
    twopi = 2.*np.pi
    # Map to 0..2pi range
    dphi = (phi1 - phi2) % twopi
    # Map pi..2pi --> -pi..0
    if is_array(dphi):
        dphi[dphi > np.pi] -= twopi
    elif dphi > np.pi:
        dphi -= twopi
    return dphi


def calc_dr(eta1, phi1, eta2, phi2):
    return np.sqrt((eta1-eta2)**2 + calc_dphi(phi1, phi2)**2)


def preselection(event):
    if len(event[b'JetsAK8.fCoordinates.fPt']) == 0:
        return False
    elif event[b'JetsAK8.fCoordinates.fPt'][0] < 550.:
        return False
    elif len(event[b'JetsAK15.fCoordinates.fPt']) < 1:
        return False
    elif not all(event[ecf][1] > 0. for ecf in [
        b'JetsAK15_ecfC2b1',
        b'JetsAK15_ecfC2b2',
        b'JetsAK15_ecfC3b1',
        b'JetsAK15_ecfC3b2',
        b'JetsAK15_ecfD2b1',
        b'JetsAK15_ecfD2b2',
        b'JetsAK15_ecfM2b1',
        b'JetsAK15_ecfM2b2',
        b'JetsAK15_ecfM3b1',
        b'JetsAK15_ecfM3b2',
        # b'JetsAK15_ecfN2b1',
        b'JetsAK15_ecfN2b2',
        # b'JetsAK15_ecfN3b1',
        b'JetsAK15_ecfN3b2'
        ]):
        return False
    elif np.sqrt(1.+event[b'MET']/event[b'JetsAK15.fCoordinates.fPt'][1]) < 1.08:
        return False
    else:
        return True


def get_subl(event):
    """
    Returns subleading jet
    """
    jets = FourVectorArray(
        event[b'JetsAK15.fCoordinates.fPt'],
        event[b'JetsAK15.fCoordinates.fEta'],
        event[b'JetsAK15.fCoordinates.fPhi'],
        event[b'JetsAK15.fCoordinates.fE'],
        ecfC2b1 = event[b'JetsAK15_ecfC2b1'],
        ecfC2b2 = event[b'JetsAK15_ecfC2b2'],
        ecfC3b1 = event[b'JetsAK15_ecfC3b1'],
        ecfC3b2 = event[b'JetsAK15_ecfC3b2'],
        ecfD2b1 = event[b'JetsAK15_ecfD2b1'],
        ecfD2b2 = event[b'JetsAK15_ecfD2b2'],
        ecfM2b1 = event[b'JetsAK15_ecfM2b1'],
        ecfM2b2 = event[b'JetsAK15_ecfM2b2'],
        ecfM3b1 = event[b'JetsAK15_ecfM3b1'],
        ecfM3b2 = event[b'JetsAK15_ecfM3b2'],
        ecfN2b1 = event[b'JetsAK15_ecfN2b1'],
        ecfN2b2 = event[b'JetsAK15_ecfN2b2'],
        ecfN3b1 = event[b'JetsAK15_ecfN3b1'],
        ecfN3b2 = event[b'JetsAK15_ecfN3b2'],
        multiplicity = event[b'JetsAK15_multiplicity'],
        girth = event[b'JetsAK15_girth'],
        ptD = event[b'JetsAK15_ptD'],
        axismajor = event[b'JetsAK15_axismajor'],
        axisminor = event[b'JetsAK15_axisminor'],
        )
    subl = jets[1]
    subl.metdphi = calc_dphi(subl.phi, event[b'METPhi'])
    return subl


def process_signal(rootfiles, outfile=None):
    n_total = 0
    n_presel = 0
    n_final = 0

    X = []

    for event in uptools.iter_events(rootfiles):
        n_total += 1
        if not preselection(event): continue
        n_presel += 1

        genparticles = FourVectorArray(
            event[b'GenParticles.fCoordinates.fPt'],
            event[b'GenParticles.fCoordinates.fEta'],
            event[b'GenParticles.fCoordinates.fPhi'],
            event[b'GenParticles.fCoordinates.fE'],
            pdgid=event[b'GenParticles_PdgId'],
            status=event[b'GenParticles_Status']
            )

        zprime = genparticles[genparticles.pdgid == 4900023]
        if len(zprime) == 0: continue
        zprime = zprime[0]

        dark_quarks = genparticles[(np.abs(genparticles.pdgid) == 4900101) & (genparticles.status == 71)]
        if len(dark_quarks) != 2: continue

        subl = get_subl(event)

        # Verify zprime and dark_quarks are within 1.5 of the jet
        if not all(calc_dr(subl.eta, subl.phi, obj.eta, obj.phi) < 1.5 for obj in [
            zprime, dark_quarks[0], dark_quarks[1]
            ]):
            continue

        n_final += 1

        X.append([
            subl.girth,
            subl.ptD,
            subl.axismajor,
            subl.axisminor,
            subl.ecfM2b1,
            subl.ecfD2b1,
            subl.ecfC2b1,
            subl.ecfN2b2,
            subl.metdphi,
            ])

    print(f'n_total: {n_total}; n_presel: {n_presel}; n_final: {n_final} ({100.*n_final/float(n_total):.2f}%)')

    if outfile is None: outfile = 'data/signal.npz'
    outdir = osp.abspath(osp.dirname(outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    print(f'Saving {n_final} entries to {outfile}')
    np.savez(outfile, X=X)


def process_bkg(rootfiles, outfile=None):
    n_total = 0
    n_presel = 0
    n_final = 0

    X = []

    for event in uptools.iter_events(rootfiles):
        n_total += 1
        if not preselection(event): continue
        n_presel += 1
        subl = get_subl(event)
        n_final += 1

        X.append([
            subl.girth,
            subl.ptD,
            subl.axismajor,
            subl.axisminor,
            subl.ecfM2b1,
            subl.ecfD2b1,
            subl.ecfC2b1,
            subl.ecfN2b2,
            subl.metdphi,
            ])

    print(f'n_total: {n_total}; n_presel: {n_presel}; n_final: {n_final} ({100.*n_final/float(n_total):.2f}%)')

    if outfile is None: outfile = 'data/bkg.npz'
    outdir = osp.abspath(osp.dirname(outfile))
    if not osp.isdir(outdir): os.makedirs(outdir)
    print(f'Saving {n_final} entries to {outfile}')
    np.savez(outfile, X=X)



def main():
    # process_signal('4881627.root')
    # process_bkg('151.root')

    process_signal('gsiftp://hepcms-gridftp.umd.edu//mnt/hadoop/cms/store/user/snabili/BKG/sig_mz250_rinv0p3_mDark20_Mar31/4881627.root')


if __name__ == '__main__':
    main()