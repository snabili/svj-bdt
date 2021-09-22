import glob
import numpy as np
import xgboost as xgb
import seutils
from combine_hists import *

try:
    import click
except ImportError:
    print('First install click:\npip install click')
    raise


def get_model(modeljson):
    model = xgb.XGBClassifier()
    model.load_model(modeljson)
    return model


@click.group()
def cli():
    pass


# Some weights

qcd_presel_eff = np.array([0.00544, 0.07959, 0.10906, 0.08919, 0.07280])
qcd_crossections = np.array([6826.0, 552.6, 156.6, 26.3, 7.5])

# ttjets order:
# Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8              831.8 * 0.105
# Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8         1.808
# Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8        0.7490
# Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8     831.8 * 0.219
# Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8  831.8 * 0.219
ttjets_presel_eff = np.array([0.00311, 0.15391, 0.30009, 0.00385, 0.00375])
ttjets_crosssections = np.array([831.8*0.105, 1.808, 0.7490, 831.8*0.219, 831.8*0.219])

# FIXME: mz350 cross section extrapolated now
mz_presel_eff = np.array([0.13847, 0.14312, 0.15442])
mz_crosssections = np.array([34820, 23430, 23430-(34820-23430)])


@cli.command()
def preselection_efficiencies_bkg():
    directories = list(sorted(glob.iglob('postbdt_npzs_Sep21_3masspoints_qcdttjets/*')))
    flength = max(len(osp.basename(d)) for d in directories)
    for directory in directories:
        npzs = glob.glob(osp.join(directory, '*.npz'))
        d = combine_npzs(npzs)
        print(
            f'{osp.basename(directory):{flength}s} :'
            f' n_total={d["n_total"]:9}, n_presel={d["n_presel"]:9},'
            f' frac={d["n_presel"]/d["n_total"]:.5f}'
            )


def combine_dirs_with_weights(directories, weights):
    ds = [combine_npzs(glob.glob(osp.join(directory, '*.npz'))) for directory in directories]
    return combine_ds_with_weights(ds, weights)


def get_combined_qcd_bkg():
    print('Reading individual qcd .npzs')
    qcd_npzs = [
        glob.glob(osp.join(directory, '*.npz'))
        for directory in [
            'postbdt_npzs/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
            'postbdt_npzs/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
            'postbdt_npzs/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
            'postbdt_npzs/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
            'postbdt_npzs/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
            ]
        ]
    print('Creating combined dict for all qcd bins')
    qcd_ds = [ combine_npzs(npzs) for npzs in qcd_npzs ]
    print('Combining qcd bins with weights')
    # qcd = combine_ds_with_weights(qcd_ds, [136.52, 278.51, 150.96, 26.24, 7.49])
    qcd = combine_ds_with_weights(qcd_ds, qcd_presel_eff*qcd_crossections)
    return qcd



def get_combined_bkg():
    print('Reading individual bkg .npzs')
    bkg_npzs = [
        glob.glob(osp.join(directory, '*.npz'))
        for directory in [
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
            'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
            ]
        ]
    print('Creating combined dict for all bkg bins')
    bkg_ds = [ combine_npzs(npzs) for npzs in bkg_npzs ]
    print('Combining bkg bins with weights')
    bkg = combine_ds_with_weights(bkg_ds, np.concatenate((qcd_presel_eff*qcd_crossections, ttjets_presel_eff*ttjets_crosssections)))
    return bkg



@cli.command()
@click.option('-o', '--rootfile', default='test.root')
def make_histograms_3masspoints_qcd_ttjets(rootfile):
    """
    With bdt version trained on only mz250 and qcd
    """
    try_import_ROOT()
    import ROOT

    qcd_dirs = [
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
        ]
    qcd_weights = qcd_crossections*qcd_presel_eff
    ttjets_dirs = [
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
        'postbdt_npzs_Sep21_3masspoints_qcdttjets/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
        ]
    ttjets_weights = ttjets_crosssections*ttjets_presel_eff

    qcd = combine_dirs_with_weights(qcd_dirs, qcd_weights)
    ttjets = combine_dirs_with_weights(ttjets_dirs, ttjets_weights)
    bkg = combine_dirs_with_weights(qcd_dirs+ttjets_dirs, np.concatenate((qcd_weights, ttjets_weights)))

    mz250 = combine_npzs(glob.glob('postbdt_npzs_Sep21_3masspoints_qcdttjets/genjetpt375_mz250_mdark10_rinv0.3/*.npz'))
    mz300 = combine_npzs(glob.glob('postbdt_npzs_Sep21_3masspoints_qcdttjets/genjetpt375_mz300_mdark10_rinv0.3/*.npz'))
    mz350 = combine_npzs(glob.glob('postbdt_npzs_Sep21_3masspoints_qcdttjets/genjetpt375_mz350_mdark10_rinv0.3/*.npz'))

    # Compute thresholds for every 10% quantile
    quantiles = np.array([i*.1 for i in range(1,10)])
    thresholds = np.quantile(bkg['score'], quantiles)

    try:
        f = ROOT.TFile.Open(rootfile, 'RECREATE')

        def dump(d, name, threshold=None, use_threshold_in_name=True):
            """
            Mini function to write a dictionary to the open root file
            """
            if threshold is not None and use_threshold_in_name:
                name += f'_{float(quantiles[thresholds == threshold]):.3f}'
            print(f'Writing {name} --> {rootfile}')
            h = make_mt_histogram(name, d['mt'], d['score'], threshold)
            h.Write()

        dump(qcd, 'qcd_mt')
        dump(ttjets, 'ttjets_mt')
        dump(bkg, 'bkg_mt')
        dump(mz250, 'mz250_mt')
        dump(mz300, 'mz300_mt')
        dump(mz350, 'mz350_mt')

        for threshold in thresholds:
            dump(qcd, 'qcd_mt', threshold)
            dump(ttjets, 'ttjets_mt', threshold)
            dump(bkg, 'bkg_mt', threshold)
            dump(mz250, 'mz250_mt', threshold)
            dump(mz300, 'mz300_mt', threshold)
            dump(mz350, 'mz350_mt', threshold)

        # For Sara
        tdir = f.mkdir('bsvj')
        tdir.cd()
        sara_threshold = thresholds[quantiles == .8]
        dump(mz250, 'SVJ_mZprime250_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz300, 'SVJ_mZprime300_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(mz350, 'SVJ_mZprime350_mDark10_rinv03_alphapeak', sara_threshold, False)
        dump(bkg, 'Bkg', sara_threshold, False)
        dump(bkg, 'data_obs', sara_threshold, False)

    finally:
        f.Close()





@cli.command()
def make_histograms_mz250_qcd():
    """
    With bdt version trained on only mz250 and qcd
    """
    try_import_ROOT()
    import ROOT

    rootfile = 'test.root'

    qcd_dirs = [
        'postbdt_npzs/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
        'postbdt_npzs/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
        'postbdt_npzs/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
        'postbdt_npzs/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
        'postbdt_npzs/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
        ]
    # qcd_weights = [136.52, 278.51, 150.96, 26.24, 7.49]
    qcd_weights = qcd_presel_eff*qcd_crossections
    qcd = combine_dirs_with_weights(qcd_dirs, qcd_weights)

    mz250 = np.load('mz250_mdark10_rinv0p3.npz')
    
    # Compute thresholds for every 10% quantile
    thresholds = np.quantile(qcd['score'], [i*.1 for i in range(1,10)])

    try:
        f = ROOT.TFile.Open(rootfile, 'RECREATE')

        def dump(d, name, threshold=None):
            """
            Mini function to write a dictionary to the open root file
            """
            if threshold is not None: name += f'_{threshold:.3f}'
            print(f'Writing {name} --> {rootfile}')
            h = make_mt_histogram(name, d['mt'], d['score'], threshold)
            h.Write()

        dump(qcd, 'qcd_mt')
        dump(mz250, 'mz250_mt')
        for threshold in thresholds:
            dump(qcd, 'qcd_mt', threshold)
            dump(mz250, 'mz250_mt', threshold)

    finally:
        f.Close()


@cli.command()
def process_mz250_locally(model):
    """
    Calculates the BDT scores for mz250 locally and dumps it to a .npz
    """
    rootfiles = seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker'
        '/genjetpt375_mz250_mdark10_rinv0.3/*.root'
        )
    get_hist_mp(model, rootfiles, 'mz250_mdark10_rinv0p3.npz')


@cli.command()
def process_qcd_locally(model):
    for qcd_dir in seutils.ls_wildcard(
        'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/*QCD_Pt*'
        ):
        print(f'Processing {qcd_dir}')
        outfile = osp.basename(qcd_dir + '.npz')
        rootfiles = seutils.ls_wildcard(osp.join(qcd_dir, '*.root'))
        get_hist_mp(model, rootfiles, outfile)



if __name__ == '__main__':
    cli()