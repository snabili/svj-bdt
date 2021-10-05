"""# submit
htcondor('request_memory', '4096MB')
import seutils, os.path as osp

root_dirs = [
    # ttjets
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018/Autumn18.TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018/Autumn18.TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018/Autumn18.TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018/Autumn18.TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018/Autumn18.TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_ttjetsAug25_year2018/Autumn18.TTJets_TuneCP5_13TeV-madgraphMLM-pythia8',
    # qcd
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/Autumn18.QCD_Pt_300to470_TuneCP5_13TeV_pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/Autumn18.QCD_Pt_470to600_TuneCP5_13TeV_pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/Autumn18.QCD_Pt_600to800_TuneCP5_13TeV_pythia8',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/Autumn18.QCD_Pt_800to1000_TuneCP5_13TeV_pythia8_ext1',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/boosted/BKG/bkg_May04_year2018/Autumn18.QCD_Pt_1000to1400_TuneCP5_13TeV_pythia8',
    # mz 250, 300, 350
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz250_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz300_mdark10_rinv0.3',
    'root://cmseos.fnal.gov//store/user/lpcdarkqcd/MCSamples_Summer21/TreeMaker/genjetpt375_mz350_mdark10_rinv0.3',
    ]

bdt_json = 'svjbdt_Sep21_fromsara_3masspoints_qcdttjets.json'

for root_dir in root_dirs:
    print('Processing {}'.format(root_dir))
    rootfiles = seutils.ls_wildcard(osp.join(root_dir, '*.root'))

    chunksize = 5
    if 'genjetpt375_mz' in root_dir:
        # Signal has less events per rootfile, do more rootfiles per job
        chunksize = 25

    for chunk in qondor.utils.chunkify(rootfiles, chunksize=chunksize):
        submit(
            rootfiles=chunk,
            bdt_json=bdt_json,
            run_env='condapack:root://cmseos.fnal.gov//store/user/klijnsma/conda-svj-bdt.tar.gz',
            transfer_files=['combine_hists.py', 'dataset.py', bdt_json],
            )
"""# endsubmit

import qondor, seutils
from combine_hists import get_hist
import xgboost as xgb
import os.path as osp, os


for rootfile in qondor.scope.rootfiles:

    try:
        seutils.cp(rootfile, 'in.root')

        model = xgb.XGBClassifier()
        model.load_model(qondor.scope.bdt_json)

        get_hist('in.root', model, 'out.npz')

        outfile = 'root://cmseos.fnal.gov//store/user/klijnsma/saras_bdt/{}/{}'.format(
            osp.basename(osp.dirname(rootfile)),
            osp.basename(rootfile).replace('.root', '.npz')
            )
        seutils.cp('out.npz', outfile)

    # except:
    #     print('Failed for rootfile ' + rootfile + ':')
    #     raise

    except Exception as e:
        print('Failed for rootfile ' + rootfile + ':')
        print(e)
        
    finally:
        if osp.isfile('out.npz'): os.remove('out.npz')
        if osp.isfile('in.root'): os.remove('in.root')
