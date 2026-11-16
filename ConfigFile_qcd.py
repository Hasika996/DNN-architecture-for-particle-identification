import glob
import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000   # progress line every 1000 events

# -1 = all events. Use e.g. 1000 for a quick test run first.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

# QCD (BACKGROUND) MiniAOD files.
# Folder name has NO spaces on purpose (rename "qcd root files" -> "qcd_root_files"):
# spaces in input paths can break cmsRun's file opening.
QCD_DIR = '/code/CMSSW_7_6_7/src/WZAnalysis/qcd_root_files'
qcd_files = sorted(glob.glob(QCD_DIR + '/*.root'))
if not qcd_files:
    raise RuntimeError('No QCD files found in ' + QCD_DIR)
print('QCD input files: %d' % len(qcd_files))

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(['file:' + f for f in qcd_files])
)

# Keep the module label "demo": the tree must stay at demo/Events.
# isTauSample=False: no tau requirement for QCD, only the common leading-jet cut.
process.demo = cms.EDAnalyzer('WZAnalyzer',
    isTauSample = cms.untracked.bool(False)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('WZAnalyzer_output_merged_qcd.root')
)

process.p = cms.Path(process.demo)