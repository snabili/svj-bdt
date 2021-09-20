import os, os.path as osp, uuid, multiprocessing, glob, shutil, logging
from time import strftime

import numpy as np
import xgboost as xgb
import seutils
import uptools
uptools.logger.setLevel(logging.WARNING)

from dataset import preselection, get_subl, calculate_mt_rt

def get_hist(rootfile, model, outfile):
    '''
    Worker function that reads a single rootfile and dumps events that
    pass the preselection to a .npz file.
    To be used in get_hist_mp.
    '''
    X = []
    X_histogram = []
    n_total = 0
    n_presel = 0
    try:
        for event in uptools.iter_events(rootfile):
            n_total += 1
            if not preselection(event): continue
            n_presel += 1
            subl = get_subl(event)
            mt, rt = calculate_mt_rt(subl, event[b'MET'], event[b'METPhi'])
            X.append([
                subl.girth, subl.ptD, subl.axismajor, subl.axisminor,
                subl.ecfM2b1, subl.ecfD2b1, subl.ecfC2b1, subl.ecfN2b2,
                subl.metdphi
                ])
            X_histogram.append([mt, rt, subl.pt, subl.energy])
    except IndexError:
        print(f'Problem with {rootfile}; saving {n_presel} good entries')
    except:
        print(f'Error processing {rootfile}; Skipping')
    if n_presel == 0:
        print(f'0/{n_total} events passed the preselection for {rootfile}')
        return
    # Get the bdt scores
    score = model.predict_proba(np.array(X))[:,1]
    # Prepare and dump to file
    print(f'Dumping {len(X)} events from {rootfile} to {outfile}')
    X_histogram = np.array(X_histogram)
    outdir = osp.dirname(outfile)
    if outdir and not osp.isdir(outdir): os.makedirs(outdir)
    np.savez(
        outfile,
        score=score,
        **{key: X_histogram[:,index] for index, key in enumerate(['mt', 'rt', 'pt', 'energy'])},
        n_total=n_total,
        n_presel=n_presel
        )

def get_hist_worker(input):
    '''Alias for get_hist that takes only 1 iterable (for mp)'''
    get_hist(*input)

def get_hist_mp(model, rootfiles, outfile, n_threads=12, keep_tmp_files=False):
    '''
    Entrypoint to read a list of rootfiles, compute the BDT scores, and combine it all
    in a single .npz file.
    Uses multiprocessing to speed things up.
    '''
    print(f'Processing {len(rootfiles)} rootfiles to {outfile}')
    tmpdir = strftime(f'TMP_%b%d_%H%M%S_{outfile}')
    os.makedirs(tmpdir)
    # Prepare input data
    data = []
    for rootfile in rootfiles:
        data.append([ rootfile, model, osp.join(tmpdir, str(uuid.uuid4())+'.npz') ])
    # Process data in multiprocessing pool
    # Every thread will dump data into tmpdir/<unique id>.npz
    pool = multiprocessing.Pool(n_threads)
    pool.map(get_hist_worker, data)
    pool.close()
    pool.join()
    # Combine the tmpdir/<unique id>.npz --> outfile and remove tmp files
    combine_npzs(tmpdir, outfile)
    if not keep_tmp_files:
        print(f'Removing {tmpdir}')
        shutil.rmtree(tmpdir)


def combine_npzs(outdir, outfile=None):
    if outfile is None: outfile = outdir.split('_', 3)[-1]
    npzs = glob.glob(osp.join(outdir, '*.npz'))
    print(f'Dumping {len(npzs)} npzs to {outfile}')
    combined = {}
    for npz in npzs:
        d = np.load(npz)
        for key, value in d.items():
            if not key in combined: combined[key] = []
            combined[key].append(value)
    # Make proper np arrays
    for key, values in combined.items():
        if len(values[0].shape) == 0:
            combined[key] = np.array(values).sum()
            print(key, combined[key])
        else:
            combined[key] = np.concatenate(values)
    np.savez(outfile, **combined)


def npz_to_TH1s(npz, label, threshold_loose=0.596, threshold_tight=0.828):
    '''Puts the array contents of a .npz into TH1s'''
    try:
        import ROOT
    except ImportError:
        print(
            'npz_to_TH1s requires ROOT to be installed too! Run:\n'
            'conda install -c conda-forge root'
            )
        raise
    from array import array
    d = np.load(npz)
    score = d['score']
    binning = array('f', [160.+8.*i for i in range(44)])
    mt = d['mt']
    mt_loose = d['mt'][score > threshold_loose]
    mt_tight = d['mt'][score > threshold_tight]
    histograms = []
    for name, values in [
        (f'{label}_mt', mt),
        (f'{label}_mt_loose', mt_loose),
        (f'{label}_mt_tight', mt_tight)
        ]:
        h = ROOT.TH1F(name, name, len(binning)-1, binning)
        ROOT.SetOwnership(h, False)
        [ h.Fill(x) for x in values ]
        histograms.append(h)
    return histograms

def npzs_to_root(npzs, labels, rootfile):
    try:
        import ROOT
    except ImportError:
        print(
            'npzs_to_root requires ROOT to be installed too! Run:\n'
            'conda install -c conda-forge root'
            )
        raise
    try:
        f = ROOT.TFile.Open(rootfile, 'RECREATE')
        for npz, label in zip(npzs, labels):
            print(f'Dumping histograms from {npz} -> {rootfile}')
            for h in npz_to_TH1s(npz, label):
                h.Write()
    finally:
        f.Close()
    

def test_npz_to_TH1s():
    # npz_to_TH1s('mz250_mdark10_rinv0p3.npz')
    npzs_to_root(['mz250_mdark10_rinv0p3.npz'], ['mz250'], 'test.root')



# ________________________________________________________
# Some tests, created during development

def test_get_hist_worker():
    model = xgb.XGBClassifier()
    model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
    rootfile = 'TREEMAKER_genjetpt375_Jul21_mz250_mdark10_rinv0.337.root'
    get_hist_worker((rootfile, model, 'out.npz'))

def test_get_hist_mp():
    model = xgb.XGBClassifier()
    model.load_model('/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json')
    rootfiles = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
        '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
        )
    outfile = 'mz250_mdark10_rinv0p3.npz'
    # get_hist_worker((rootfiles[1], model, 'out.npz'))
    get_hist_mp(model, rootfiles, outfile)
# ________________________________________________________


def process_mz250(model):
    rootfiles = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
        '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
        )
    get_hist_mp(model, rootfiles, 'mz250_mdark10_rinv0p3.npz')

def process_qcd(model):
    for qcd_dir in seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/*QCD_Pt*'
        ):
        print(f'Processing {qcd_dir}')
        outfile = osp.basename(qcd_dir + '.npz')
        rootfiles = seutils.ls_wildcard(osp.join(qcd_dir, '*.root'))
        get_hist_mp(model, rootfiles, outfile)


def cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['mz250', 'qcd'], type=str)
    parser.add_argument(
        '-m', '--modeljson', type=str,
        default='/Users/klijnsma/work/svj/bdt/svjbdt_Aug02.json'
        )
    args = parser.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.modeljson)

    if args.action == 'mz250':
        process_mz250(model)
    elif args.action == 'qcd':
        process_qcd(model)


if __name__ == '__main__':
    # test_npz_to_TH1s()
    cli()
    # test_get_hist_worker()
    # test_get_hist_mp()
    # combine_npzs('TMP_Sep14_133719_mz250_mdark10_rinv0p3.npz')