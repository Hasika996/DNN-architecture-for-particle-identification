import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1000   # progress line every 1000 events

# -1 = all events. Use e.g. 1000 for a quick test run first.
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    # TAU (SIGNAL) MiniAOD files
    fileNames = cms.untracked.vstring(
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/0A0AA368-CBEB-E511-A406-0D23B6AFA6B4.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/94E4A5E3-92EC-E511-9E53-002590908226.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/929D9D11-29ED-E511-8E3A-0025904C7F72.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/001268DD-CAEB-E511-AC7F-842B2B5C2299.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/007555B7-7EEF-E511-8937-00266CF9AE10.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/864492DA-A8EC-E511-B2FF-00259073E52A.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/A021D37C-CAEB-E511-BBD4-0025904A8ECE.root',
        'file:/code/CMSSW_7_6_7/src/WZAnalysis/WZAnalyzer/python/964752C9-9BEC-E511-9087-A0369F7FC540.root'
    )
)

# Keep the module label "demo": the tree must stay at demo/Events.
# isTauSample=True switches on the reco-tau requirement (>= 2 reco taus in the leading jet).
process.demo = cms.EDAnalyzer('WZAnalyzer',
    isTauSample = cms.untracked.bool(True)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('WZAnalyzer_output_merged_tau.root')
)

process.p = cms.Path(process.demo)