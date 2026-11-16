#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdio>
#include <utility>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "TTree.h"
#include "TH1D.h"

// FastJet: used to re-cluster the leading jet's own PF constituents with
// the kT algorithm into a fixed number of *exclusive* subjets, which is
// how lambda_J and epsilon_J (Eq. 3 of the paper) are defined, and to
// compute the AK4 N-subjettiness (tau1, tau31).
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

// these things below are the working points for b-tag
//loose wp - 10% mistag rate, medium = 1% misrag rate, tight = 0.1% mistag rate
// got these values from CSVv2 CERN twiki page. 
static const float CSVv2_LOOSE_WP  = 0.605;
static const float CSVv2_MEDIUM_WP = 0.890;
static const float CSVv2_TIGHT_WP  = 0.970;

// n_t and n_f from the paper (Sec. 3.3, following Ref.[16,17]): n_t is
// the number of exclusive kT subjets used to build lambda_J (Eq. 3-4),
// n_f is the (smaller) number used for the energy-energy correlation
// epsilon_J.
static const unsigned int NT_SUBJETS = 5;
static const unsigned int NF_SUBJETS = 3;

// ----------------------------------------------------------------------
// COMMON SELECTION -- identical for every variable and for both samples
// ----------------------------------------------------------------------
// Leading AK4 jet (slimmedJets): pT > 50 GeV and |eta| < 2.4. The leading
// jet is the highest-pT jet among those passing BOTH cuts.
static const double LEADING_JET_MIN_PT  = 50.0;
static const double LEADING_JET_MAX_ETA = 2.4;
// Tau (signal) sample ONLY: count RECONSTRUCTED taus (slimmedTaus, passing
// decayModeFinding) within DeltaR < TAU_JET_MAX_DR of the leading jet axis,
// and require at least MIN_TAUS_IN_JET of them. QCD gets NO tau selection and
// no tau matching of any kind.
// Keep MIN_TAUS_IN_JET = 1: the di-tau selection below (one tau in each of the
// two leading jets) already implies a tau in the leading jet, and a value of 2
// would instead demand two taus inside the SAME jet.
static const double TAU_JET_MAX_DR  = 0.4;
static const int    MIN_TAUS_IN_JET = 1;

// Cutflow histogram bins (1-based). Counts are RAW event counts (no genWeight),
// so the histograms from several files simply add up with hadd.
//   CF_TAU_MATCHED, CF_TWO_JETS and CF_TAU_PAIR are filled ONLY for the tau
//   sample (they are selection steps for signal). For QCD these three bins
//   stay at 0 -- QCD has no tau steps. The bin layout is unchanged so that
//   outputs from older and newer runs can still be hadd'ed together.
enum CutflowBin {
    CF_ALL = 1,        // every event that enters analyze()
    CF_VERTEX,         // has a primary vertex
    CF_JET_PT,         // >= 1 jet with pT > LEADING_JET_MIN_PT (any eta)
    CF_LEADING_JET,    // >= 1 jet with pT > LEADING_JET_MIN_PT and |eta| < LEADING_JET_MAX_ETA
    CF_TAU_MATCHED,    // (tau sample only) >= MIN_TAUS_IN_JET reco tau(s) within dR < TAU_JET_MAX_DR of the leading jet
    CF_TWO_JETS,       // (tau sample only) >= 2 jets passing the same pT and |eta| cuts (j1 = leading, j2 = second)
    CF_TAU_PAIR,       // (tau sample only) two DIFFERENT reco taus, one within dR < TAU_JET_MAX_DR of j1 and one of j2
    CF_FILLED,         // event written to the tree (final selection)
    CF_NBINS = CF_FILLED
};

// ----------------------------------------------------------------------
// Di-tau matching (gg -> tau tau): true if there are two DIFFERENT taus, one
// within dR < maxDR of jet 1 and the other within dR < maxDR of jet 2. Trying
// every ordered pair (a -> j1, b -> j2) covers both pairings of
//   ( dR(tau1,j1)<maxDR && dR(tau2,j2)<maxDR ) || ( dR(tau1,j2)<maxDR && dR(tau2,j1)<maxDR )
// regardless of how the taus are labelled; a != b stops a single tau from
// counting for both jets when the two jets overlap.
// ----------------------------------------------------------------------
static bool tauPairMatchesJets(const std::vector<std::pair<double, double> >& tauEtaPhi,
                               double j1Eta, double j1Phi,
                               double j2Eta, double j2Phi, double maxDR)
{
    const size_t n = tauEtaPhi.size();
    for (size_t a = 0; a < n; ++a) {
        if (!(reco::deltaR(tauEtaPhi[a].first, tauEtaPhi[a].second, j1Eta, j1Phi) < maxDR)) continue;
        for (size_t b = 0; b < n; ++b) {
            if (b == a) continue;
            if (reco::deltaR(tauEtaPhi[b].first, tauEtaPhi[b].second, j2Eta, j2Phi) < maxDR) return true;
        }
    }
    return false;
}

// --- N-subjettiness (tau1/tau3, AK4) support -----------------------------
// dR between two fastjet::PseudoJets, needed by calculateNSubjettiness()
// below (kept separate from reco::deltaR, which only takes reco/pat types).
static double fastjetDeltaR(const fastjet::PseudoJet& p1,
                            const fastjet::PseudoJet& p2)
{
    double dEta = p1.eta() - p2.eta();
    double dPhi = p1.phi() - p2.phi();
    while (dPhi > M_PI)  dPhi -= 2.0 * M_PI;
    while (dPhi < -M_PI) dPhi += 2.0 * M_PI;
    return std::sqrt(dEta*dEta + dPhi*dPhi);
}

// Reconstructed jet constituents only (no Gen particles). Computes
// N-subjettiness tau_N via exclusive-kt subjet axes, R=0.4 (AK4 path).
// AK8 tau1/tau3 are read directly from the stored NjettinessAK8 userFloats
// instead (one-pass-minimized/WTA axes), so this function is only used
// for the AK4 leading jet.
static double calculateNSubjettiness(const pat::Jet& jet, int N, double R = 0.4)
{
    if (N < 1) return -1.0;

    std::vector<fastjet::PseudoJet> constituents;
    for (size_t i = 0; i < jet.numberOfDaughters(); ++i) {
        const reco::Candidate* cand = jet.daughter(i);
        if (!cand || cand->pt() <= 0.0) continue;
        constituents.emplace_back(cand->px(), cand->py(), cand->pz(), cand->energy());
    }

    if (constituents.size() < static_cast<size_t>(N)) return -1.0;

    fastjet::JetDefinition ktDefinition(fastjet::kt_algorithm, R, fastjet::E_scheme);
    fastjet::ClusterSequence cs(constituents, ktDefinition);

    std::vector<fastjet::PseudoJet> axes;
    try {
        axes = cs.exclusive_jets(N);
    } catch (...) {
        return -1.0;
    }

    if (axes.size() != static_cast<size_t>(N)) return -1.0;

    double numerator = 0.0, denominator = 0.0;
    for (const auto& particle : constituents) {
        double minDR = 1.0e10;
        for (const auto& axis : axes) {
            double dR = fastjetDeltaR(particle, axis);
            if (dR < minDR) minDR = dR;
        }
        numerator += particle.perp() * minDR;
        denominator += particle.perp() * R;
    }

    if (denominator <= 0.0) return -1.0;
    return numerator / denominator;
}

class WZAnalyzer : public edm::EDAnalyzer {
public:
    explicit WZAnalyzer(const edm::ParameterSet&);
    ~WZAnalyzer();
private:
    virtual void beginJob() override;
    virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() override;

    // Re-cluster a jet's PF constituents with the kT algorithm and return
    // exactly nSubjets *exclusive* subjets (Eq. 3-4 of the paper),
    // sorted by descending pT (subjet 0 = hardest). Returns false, and
    // leaves subjets untouched, if the jet has fewer than nSubjets
    // constituents (can't form that many exclusive subjets).
    bool getExclusiveSubjets(const pat::Jet& jet, unsigned int nSubjets,
                              std::vector<fastjet::PseudoJet>& subjets);

    edm::EDGetTokenT<double>                          rhoToken_;
    edm::EDGetTokenT<std::vector<reco::Vertex>>       vtxToken_;
    edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  puToken_;
    edm::EDGetTokenT<pat::MuonCollection>             muonToken_;
    edm::EDGetTokenT<pat::ElectronCollection>         electronToken_;
    edm::EDGetTokenT<pat::METCollection>              metToken_;
    edm::EDGetTokenT<pat::JetCollection>              jetToken_;
    edm::EDGetTokenT<pat::JetCollection>              jetAK8Token_;
    edm::EDGetTokenT<pat::PhotonCollection>          photonToken_;
    edm::EDGetTokenT<pat::TauCollection>              tauToken_;
    edm::EDGetTokenT<pat::PackedCandidateCollection>  packedCandidateToken_;
    edm::EDGetTokenT<std::vector<reco::GenParticle>>  genToken_;
    edm::EDGetTokenT<edm::TriggerResults>             metFilterToken_;
    edm::EDGetTokenT<GenEventInfoProduct>             genInfoToken_;

    TTree* tree_;
    bool isTauSample_;
    unsigned int run_, lumi_;
    unsigned long long event_;
    int nPV_; float rho_; int nTruePU_;

    // Cutflow histogram (see CutflowBin). Saved in the output file and printed in endJob().
    TH1D* h_cutflow_;

    int nMuons_;
    std::vector<float> mu_pt_, mu_eta_, mu_phi_, mu_energy_, mu_mass_, mu_charge_, mu_relIso_, mu_dxy_, mu_dz_;
    std::vector<int> mu_isLoose_, mu_isMedium_, mu_isTight_;

    int nElectrons_;
    std::vector<float> el_pt_, el_eta_, el_phi_, el_energy_, el_mass_, el_charge_, el_relIso_, el_dxy_, el_dz_;

    float met_pt_, met_phi_, met_px_, met_py_, met_sumet_;

    // since we have used SlimmedJets, these jets are automatically AK4 and we can also use b-tagging information from these directly
    int nJets_;
    std::vector<float> jet_pt_, jet_eta_, jet_phi_, jet_energy_, jet_mass_, jet_csv_;
    std::vector<int> jet_hadFlav_;
// jet_csv_ gives the b0tagging discriminating value for each jet.  (csvv2 = combined secondary vertex version 2)

    // Photons
    int nPhotons_;
    std::vector<float> pho_pt_, pho_eta_, pho_phi_, pho_energy_;
    std::vector<float> pho_sigmaIetaIeta_, pho_hOverE_, pho_relIso_;
    std::vector<int>   pho_passLooseId_;

    int nGenLep_;
    std::vector<float> gen_pt_, gen_eta_, gen_phi_, gen_mass_;
    std::vector<int> gen_pdgId_, gen_status_, gen_motherPdgId_;

    
    int nGenPhotons_;
    std::vector<float> gen_pho_pt_, gen_pho_eta_, gen_pho_phi_, gen_pho_energy_;
    std::vector<int>   gen_pho_status_, gen_pho_isPrompt_, gen_pho_motherPdgId_;

    // matching between reco photons and gen photons    
    std::vector<int> pho_isPromptMatched_, pho_matchedGenStatus_;

    // Jet b-tag working-point flags 
    std::vector<int> jet_btagLoose_, jet_btagMedium_, jet_btagTight_;

    // Fig. 1 (left) histogram: log10(theta_J) for the leading jet of this
    // sample. Booked with TFileService so it lands directly in the output
    // ROOT file alongside the tree, ready to be drawn/overlaid like the
    // paper's Fig. 1.
    TH1D* h_logThetaJ_;

    // Fig. 2 (left) histogram: lambda_J = log(1 - pT_0/pT_J), Eq. (3),
    // built from the leading jet's nt=5 exclusive kT subjets.
    // Fig. 2 (right) histogram: epsilon_J, the energy-energy correlation
    // among the leading jet's nf=3 exclusive kT subjets, also Eq. (3).
    // Both booked with TFileService, same as h_logThetaJ_.
    TH1D* h_lambdaJ_;
    TH1D* h_epsilonJ_;

    // Histograms for the additional jet-shape / N-subjettiness variables.
    TH1D* h_nTracks_;
    TH1D* h_logTau1_;
    TH1D* h_tau31_;
    TH1D* h_logTau1AK8_;
    TH1D* h_tau31AK8_;

    // Variables for the leading-jet hadronic energy fraction used in Fig. 1
    int leadingJetIndex_;
    int leadingJetNConstituents_;
    int leadingJetNHCALConstituents_;
    float leadingJetPt_, leadingJetEta_, leadingJetPhi_;
    float leadingJetE_, leadingJetEJ_, leadingJetEHCAL_;
    float leadingJetThetaJ_, leadingJetLogThetaJ_;

    // Leading-jet substructure variables for Fig. 2.
    float leadingJetLambdaJ_;
    float leadingJetEpsilonJ_;

    // Number of reco taus (decayModeFinding) within dR < TAU_JET_MAX_DR of
    // the leading jet. Computed and required to be >= MIN_TAUS_IN_JET for the
    // TAU sample only; for QCD no tau matching is done and it stays at -1.
    // NOT a BDT feature: it is used to select the signal, so training on it
    // would leak the label.
    int nTausInJet_;

    // Di-tau selection diagnostics (see tauPairMatchesJets). Tau sample only;
    // both stay at -1 for QCD (no tau selection is done there).
    //   secondJetIndex_ : index of j2 in the jet_* vectors (-1 if not evaluated / fewer than 2 selected jets)
    //   diTauPair_      : 1 = one tau in each of j1, j2; 0 = not; -1 = not evaluated
    // NOT BDT features: they define the signal selection, so training on them leaks the label.
    int secondJetIndex_;
    int diTauPair_;

    // Additional per-event jet-shape / N-subjettiness variables (AK4 + AK8),
    // computed from the SAME leading AK4 jet (bestJetPtr) selected above.
    int nTracks_;
    float tau1_, tau31_, tau1AK8_, tau31AK8_;

    // Generator event weight 
    float genWeight_;
     // it is the number assigned by MC generator to each event. in LO samples, it is usually +/- 1 . 
     // In NLO samples, it can be it can be positive/negative and varies by event
     // when we calculate cross-sections or make histograms,we need to multiply it by genWeight
     // here we always fill histogrrams with genWeight_ = iEvent.genWeight() ;  // this is the genWeight for each event

    int flag_goodVertices_, flag_HBHENoiseFilter_, flag_HBHENoiseIsoFilter_,
        flag_EcalDeadCellTriggerPrimitiveFilter_, flag_eeBadScFilter_,
        flag_globalTightHalo_;
        // these are data quality filters to remove bad events caused by detector problems. 
        // in real data, we must require most of these flags to be 1 for that event to pass. 
};

WZAnalyzer::WZAnalyzer(const edm::ParameterSet& iConfig) :
    rhoToken_     (consumes<double>                         (edm::InputTag("fixedGridRhoFastjetAll"))),
    vtxToken_     (consumes<std::vector<reco::Vertex>>      (edm::InputTag("offlineSlimmedPrimaryVertices"))),
    puToken_      (consumes<std::vector<PileupSummaryInfo>> (edm::InputTag("slimmedAddPileupInfo"))),
    muonToken_    (consumes<pat::MuonCollection>            (edm::InputTag("slimmedMuons"))),
    electronToken_(consumes<pat::ElectronCollection>        (edm::InputTag("slimmedElectrons"))),
    metToken_     (consumes<pat::METCollection>             (edm::InputTag("slimmedMETs"))),
    jetToken_     (consumes<pat::JetCollection>             (edm::InputTag("slimmedJets"))),
    jetAK8Token_  (consumes<pat::JetCollection>             (edm::InputTag("slimmedJetsAK8"))),
    photonToken_  (consumes<pat::PhotonCollection>           (edm::InputTag("slimmedPhotons"))),
    tauToken_     (consumes<pat::TauCollection>              (edm::InputTag("slimmedTaus"))),
    packedCandidateToken_(consumes<pat::PackedCandidateCollection>(edm::InputTag("packedPFCandidates"))),
    genToken_     (consumes<std::vector<reco::GenParticle>> (edm::InputTag("prunedGenParticles"))),
    metFilterToken_(consumes<edm::TriggerResults>           (edm::InputTag("TriggerResults","","PAT"))),
    genInfoToken_  (consumes<GenEventInfoProduct>           (edm::InputTag("generator"))),
    isTauSample_   (iConfig.getUntrackedParameter<bool>("isTauSample", false))
{}

WZAnalyzer::~WZAnalyzer() {}

void WZAnalyzer::beginJob() {
    edm::Service<TFileService> fs;

    // Cutflow: one bin per selection step, labelled so it can be read straight from ROOT.
    h_cutflow_ = fs->make<TH1D>("h_cutflow",
        "Cutflow (raw event counts);selection step;events",
        CF_NBINS, 0.5, CF_NBINS + 0.5);
    {
        std::ostringstream jetPt, lead, tauM, twoJ, pairL;
        jetPt << ">=1 jet with pT>" << LEADING_JET_MIN_PT;
        lead  << "Leading jet pT>" << LEADING_JET_MIN_PT << ", |eta|<" << LEADING_JET_MAX_ETA;
        tauM  << ">=" << MIN_TAUS_IN_JET << " reco tau in jet (dR<" << TAU_JET_MAX_DR << ")";
        twoJ  << ">=2 jets pT>" << LEADING_JET_MIN_PT << ", |eta|<" << LEADING_JET_MAX_ETA;
        pairL << "1 tau in each of j1,j2 (dR<" << TAU_JET_MAX_DR << ")";
        h_cutflow_->GetXaxis()->SetBinLabel(CF_ALL,          "All events");
        h_cutflow_->GetXaxis()->SetBinLabel(CF_VERTEX,       "Primary vertex");
        h_cutflow_->GetXaxis()->SetBinLabel(CF_JET_PT,       jetPt.str().c_str());
        h_cutflow_->GetXaxis()->SetBinLabel(CF_LEADING_JET,  lead.str().c_str());
        h_cutflow_->GetXaxis()->SetBinLabel(CF_TAU_MATCHED,  tauM.str().c_str());
        h_cutflow_->GetXaxis()->SetBinLabel(CF_TWO_JETS,     twoJ.str().c_str());
        h_cutflow_->GetXaxis()->SetBinLabel(CF_TAU_PAIR,     pairL.str().c_str());
        h_cutflow_->GetXaxis()->SetBinLabel(CF_FILLED,       "Final (written to tree)");
    }

    // ----------------------------------------------------------------
    // Fig. 1 (left) of arXiv:1707.07084: log10(theta_J) for the leading
    // jet, theta_J defined via Eq. (2):
    //     theta_J = (1/E_J) * sum_{i in J, HCAL} E_i ,  E_J = sum_{i in J} E_i
    // The paper's axis range is roughly log(theta_J) in [-3, 0]; we use
    // 60 bins (bin width 0.05) over that same range. Since the paper's
    // plots are normalized to unit area ("a.u."), do Scale(1.0/Integral())
    // on this histogram before drawing/overlaying it with the other
    // standard-object samples (photon/electron/QCD), exactly as in Fig. 1.
    // ----------------------------------------------------------------
    h_logThetaJ_ = fs->make<TH1D>("h_logThetaJ",
        "Leading-jet log_{10}(#theta_{J});log_{10}(#theta_{J});a.u.",
        60, -3.0, 0.0);

    // ----------------------------------------------------------------
    // Fig. 2 (left) of arXiv:1707.07084: lambda_J, Eq. (3), for the
    // leading jet, using nt=5 exclusive kT subjets. From the discussion
    // around Eq. (4), a single-prong (pencil-like) jet drives lambda_J
    // toward -infinity while a very democratic n-prong jet approaches 0
    // (e.g. lambda ~ -0.30 for two roughly equal prongs). The paper's
    // plot spans roughly [-6, 0]; we use 60 bins over that range.
    // Fig. 2 (right): epsilon_J, also Eq. (3), using nf=3 subjets. The
    // paper's plot spans roughly [0, 0.35].
    // As with h_logThetaJ_, these are meant to be Scale(1/Integral())'d
    // ("a.u.") when drawn/overlaid across the standard-object samples.
    // ----------------------------------------------------------------
    h_lambdaJ_ = fs->make<TH1D>("h_lambdaJ",
        "Leading-jet #lambda_{J};#lambda_{J};a.u.",
        60, -6.0, 0.0);
    h_epsilonJ_ = fs->make<TH1D>("h_epsilonJ",
        "Leading-jet #epsilon_{J};#epsilon_{J};a.u.",
        60, 0.0, 0.35);

    // Additional jet-shape / N-subjettiness histograms, same leading jet.
    h_nTracks_ = fs->make<TH1D>("h_nTracks",
        "Number of tracks in leading jet;N_{T};a.u.", 30, 0.0, 30.0);
    h_logTau1_ = fs->make<TH1D>("h_logTau1",
        "N-subjettiness (AK4, custom);log(#tau_{1});a.u.", 60, -3.0, 0.0);
    h_tau31_ = fs->make<TH1D>("h_tau31",
        "N-subjettiness ratio (AK4, custom);#tau_{31}=#tau_{3}/#tau_{1};a.u.", 60, 0.0, 0.9);
    h_logTau1AK8_ = fs->make<TH1D>("h_logTau1AK8",
        "N-subjettiness (AK8, NjettinessAK8);log(#tau_{1});a.u.", 60, -3.0, 0.0);
    h_tau31AK8_ = fs->make<TH1D>("h_tau31AK8",
        "N-subjettiness ratio (AK8, NjettinessAK8);#tau_{31}=#tau_{3}/#tau_{1};a.u.", 60, 0.0, 0.9);

    tree_ = fs->make<TTree>("Events", "WZ Analysis Tree");
    tree_->Branch("run",         &run_);
    tree_->Branch("lumi",        &lumi_);
    tree_->Branch("event",       &event_);
    tree_->Branch("nPV",         &nPV_);
    tree_->Branch("rho",         &rho_);
    tree_->Branch("nTruePU",     &nTruePU_);

    tree_->Branch("nMuons",      &nMuons_);
    tree_->Branch("mu_pt",       &mu_pt_);
    tree_->Branch("mu_eta",      &mu_eta_);
    tree_->Branch("mu_phi",      &mu_phi_);
    tree_->Branch("mu_energy",   &mu_energy_);
    tree_->Branch("mu_mass",     &mu_mass_);
    tree_->Branch("mu_charge",   &mu_charge_);
    tree_->Branch("mu_relIso",   &mu_relIso_);
    tree_->Branch("mu_dxy",      &mu_dxy_);
    tree_->Branch("mu_dz",       &mu_dz_);
    tree_->Branch("mu_isLoose",  &mu_isLoose_);
    tree_->Branch("mu_isMedium", &mu_isMedium_);
    tree_->Branch("mu_isTight",  &mu_isTight_);

    tree_->Branch("nElectrons",  &nElectrons_);
    tree_->Branch("el_pt",       &el_pt_);
    tree_->Branch("el_eta",      &el_eta_);
    tree_->Branch("el_phi",      &el_phi_);
    tree_->Branch("el_energy",   &el_energy_);
    tree_->Branch("el_mass",     &el_mass_);
    tree_->Branch("el_charge",   &el_charge_);
    tree_->Branch("el_relIso",   &el_relIso_);
    tree_->Branch("el_dxy",      &el_dxy_);
    tree_->Branch("el_dz",       &el_dz_);

    tree_->Branch("met_pt",      &met_pt_);
    tree_->Branch("met_phi",     &met_phi_);
    tree_->Branch("met_px",      &met_px_);
    tree_->Branch("met_py",      &met_py_);
    tree_->Branch("met_sumet",   &met_sumet_);

    tree_->Branch("nJets",       &nJets_);
    tree_->Branch("jet_pt",      &jet_pt_);
    tree_->Branch("jet_eta",     &jet_eta_);
    tree_->Branch("jet_phi",     &jet_phi_);
    tree_->Branch("jet_energy",  &jet_energy_);
    tree_->Branch("jet_mass",    &jet_mass_);
    tree_->Branch("jet_csv",     &jet_csv_);
    tree_->Branch("jet_hadFlav", &jet_hadFlav_);

    tree_->Branch("nPhotons",          &nPhotons_);
    tree_->Branch("pho_pt",            &pho_pt_);
    tree_->Branch("pho_eta",           &pho_eta_);
    tree_->Branch("pho_phi",           &pho_phi_);
    tree_->Branch("pho_energy",        &pho_energy_);
    tree_->Branch("pho_sigmaIetaIeta", &pho_sigmaIetaIeta_);
    tree_->Branch("pho_hOverE",        &pho_hOverE_);
    tree_->Branch("pho_relIso",        &pho_relIso_);
    tree_->Branch("pho_passLooseId",   &pho_passLooseId_);
    tree_->Branch("nGenLep",          &nGenLep_);
    tree_->Branch("gen_pt",           &gen_pt_);
    tree_->Branch("gen_eta",          &gen_eta_);
    tree_->Branch("gen_phi",          &gen_phi_);
    tree_->Branch("gen_mass",         &gen_mass_);
    tree_->Branch("gen_pdgId",        &gen_pdgId_);
    tree_->Branch("gen_status",       &gen_status_);
    tree_->Branch("gen_motherPdgId",  &gen_motherPdgId_);

    
    tree_->Branch("nGenPhotons",        &nGenPhotons_);
    tree_->Branch("gen_pho_pt",         &gen_pho_pt_);
    tree_->Branch("gen_pho_eta",        &gen_pho_eta_);
    tree_->Branch("gen_pho_phi",        &gen_pho_phi_);
    tree_->Branch("gen_pho_energy",     &gen_pho_energy_);
    tree_->Branch("gen_pho_status",     &gen_pho_status_);
    tree_->Branch("gen_pho_isPrompt",   &gen_pho_isPrompt_);
    tree_->Branch("gen_pho_motherPdgId",&gen_pho_motherPdgId_);

    tree_->Branch("pho_isPromptMatched",  &pho_isPromptMatched_);
    tree_->Branch("pho_matchedGenStatus", &pho_matchedGenStatus_);

    tree_->Branch("jet_btagLoose",  &jet_btagLoose_);
    tree_->Branch("jet_btagMedium", &jet_btagMedium_);
    tree_->Branch("jet_btagTight",  &jet_btagTight_);

    // Leading-jet variables for the paper's Fig. 1 left panel.
    // EJ is the sum of constituent energies; EHCAL is the sum of the
    // HCAL-energy fractions carried by the jet constituents.
    tree_->Branch("leadingJetIndex",          &leadingJetIndex_);
    tree_->Branch("leadingJetNConstituents",  &leadingJetNConstituents_);
    tree_->Branch("leadingJetNHCALConstituents", &leadingJetNHCALConstituents_);
    tree_->Branch("leadingJetPt",             &leadingJetPt_);
    tree_->Branch("leadingJetEta",            &leadingJetEta_);
    tree_->Branch("leadingJetPhi",            &leadingJetPhi_);
    tree_->Branch("leadingJetE",              &leadingJetE_);
    tree_->Branch("leadingJetEJ",             &leadingJetEJ_);
    tree_->Branch("leadingJetEHCAL",          &leadingJetEHCAL_);
    tree_->Branch("leadingJetThetaJ",         &leadingJetThetaJ_);
    tree_->Branch("leadingJetLogThetaJ",      &leadingJetLogThetaJ_);

    // Fig. 2 substructure variables (Eq. 3): lambda_J from nt=5 subjets,
    // epsilon_J from nf=3 subjets.
    tree_->Branch("leadingJetLambdaJ",        &leadingJetLambdaJ_);
    tree_->Branch("leadingJetEpsilonJ",       &leadingJetEpsilonJ_);

    // Diagnostic only (see comment at the declaration) -- do not train on it.
    // Tau sample only; -1 for QCD.
    tree_->Branch("nTausInJet",               &nTausInJet_);
    tree_->Branch("secondJetIndex",           &secondJetIndex_);
    tree_->Branch("diTauPair",                &diTauPair_);

    // Additional jet-shape / N-subjettiness branches (same leading jet).
    tree_->Branch("nTracks",  &nTracks_);
    tree_->Branch("tau1",     &tau1_);
    tree_->Branch("tau31",    &tau31_);
    tree_->Branch("tau1AK8",  &tau1AK8_);
    tree_->Branch("tau31AK8", &tau31AK8_);

    tree_->Branch("genWeight", &genWeight_);

    tree_->Branch("Flag_goodVertices",                      &flag_goodVertices_);
    tree_->Branch("Flag_HBHENoiseFilter",                   &flag_HBHENoiseFilter_);
    tree_->Branch("Flag_HBHENoiseIsoFilter",                &flag_HBHENoiseIsoFilter_);
    tree_->Branch("Flag_EcalDeadCellTriggerPrimitiveFilter",&flag_EcalDeadCellTriggerPrimitiveFilter_);
    tree_->Branch("Flag_eeBadScFilter",                     &flag_eeBadScFilter_);
    tree_->Branch("Flag_globalTightHalo",                   &flag_globalTightHalo_);
}

void WZAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
    mu_pt_.clear(); mu_eta_.clear(); mu_phi_.clear(); mu_energy_.clear();
    mu_mass_.clear(); mu_charge_.clear(); mu_relIso_.clear();
    mu_dxy_.clear(); mu_dz_.clear();
    mu_isLoose_.clear(); mu_isMedium_.clear(); mu_isTight_.clear();
    el_pt_.clear(); el_eta_.clear(); el_phi_.clear(); el_energy_.clear();
    el_mass_.clear(); el_charge_.clear(); el_relIso_.clear();
    el_dxy_.clear(); el_dz_.clear();
    jet_pt_.clear(); jet_eta_.clear(); jet_phi_.clear(); jet_energy_.clear();
    jet_mass_.clear(); jet_csv_.clear(); jet_hadFlav_.clear();
    gen_pt_.clear(); gen_eta_.clear(); gen_phi_.clear(); gen_mass_.clear();
    gen_pdgId_.clear(); gen_status_.clear(); gen_motherPdgId_.clear();
    pho_pt_.clear(); pho_eta_.clear(); pho_phi_.clear(); pho_energy_.clear();
    pho_sigmaIetaIeta_.clear(); pho_hOverE_.clear(); pho_relIso_.clear();
    pho_passLooseId_.clear();
    gen_pho_pt_.clear(); gen_pho_eta_.clear(); gen_pho_phi_.clear(); gen_pho_energy_.clear();
    gen_pho_status_.clear(); gen_pho_isPrompt_.clear(); gen_pho_motherPdgId_.clear();
    pho_isPromptMatched_.clear(); pho_matchedGenStatus_.clear();
    jet_btagLoose_.clear(); jet_btagMedium_.clear(); jet_btagTight_.clear();

    // Sentinels for "variable could not be computed". The BDT script should
    // turn these into NaN (and/or *_failed flags) rather than dropping events.
    leadingJetIndex_ = -1;
    leadingJetNConstituents_ = 0;
    leadingJetNHCALConstituents_ = 0;
    leadingJetPt_ = -1.0f;
    leadingJetEta_ = 0.0f;
    leadingJetPhi_ = 0.0f;
    leadingJetE_ = -1.0f;
    leadingJetEJ_ = -1.0f;
    leadingJetEHCAL_ = -1.0f;
    leadingJetThetaJ_ = -1.0f;
    leadingJetLogThetaJ_ = -99.0f;
    leadingJetLambdaJ_ = -99.0f;
    leadingJetEpsilonJ_ = -1.0f;

    nTausInJet_ = -1;
    secondJetIndex_ = -1;
    diTauPair_ = -1;

    nTracks_  = -1;
    tau1_     = -999.0f;
    tau31_    = -999.0f;
    tau1AK8_  = -999.0f;
    tau31AK8_ = -999.0f;

    h_cutflow_->Fill(CF_ALL);

    run_   = iEvent.id().run();
    lumi_  = iEvent.id().luminosityBlock();
    event_ = iEvent.id().event();

    edm::Handle<double> rhoHandle;
    iEvent.getByToken(rhoToken_, rhoHandle);
    rho_ = rhoHandle.isValid() ? (float)(*rhoHandle) : 0.0f;
    //it measures how many pileup interactions are in the event. 
    //it is used to correct for isolation and jet energy corrections.
// it is calculated using the FastJet algorithm, which clusters all the particles in the event and calculates the median energy density in the event.
// its main use is to correct for the extra energy deposited in the detector due to pileup interactions, which can affect the measurement of jets and missing transverse energy.
// without rho correction, the isolation and jet energy corrections would be biased by the extra energy from pileup interactions, leading to incorrect measurements of the physics objects in the event.
    edm::Handle<std::vector<reco::Vertex>> vertices;
    iEvent.getByToken(vtxToken_, vertices);
    nPV_ = vertices.isValid() ? (int)vertices->size() : 0;
    // No primary vertex -> event dropped (same for every variable).
    if (!vertices.isValid() || vertices->empty()) return;
    h_cutflow_->Fill(CF_VERTEX);
    const reco::Vertex& pv = vertices->at(0);

    nTruePU_ = -1;
    edm::Handle<std::vector<PileupSummaryInfo>> puInfo;
    iEvent.getByToken(puToken_, puInfo);
    if (puInfo.isValid()) {
    for (const auto& pu : *puInfo) {
        if (pu.getBunchCrossing() == 0) { nTruePU_ = (int)pu.getTrueNumInteractions(); break; }
    }
    }

    edm::Handle<pat::MuonCollection> muons;
    iEvent.getByToken(muonToken_, muons);
    nMuons_ = 0;
    if (muons.isValid()) for (const pat::Muon& mu : *muons) {
        if (mu.pt() < 10.0 || fabs(mu.eta()) > 2.4) continue;
        mu_pt_.push_back(mu.pt());
        mu_eta_.push_back(mu.eta());
        mu_phi_.push_back(mu.phi());
        mu_energy_.push_back(mu.energy());
        mu_mass_.push_back(mu.mass());
        mu_charge_.push_back(mu.charge());
        float chIso = mu.pfIsolationR04().sumChargedHadronPt;
        float nhIso = mu.pfIsolationR04().sumNeutralHadronEt;
        float phIso = mu.pfIsolationR04().sumPhotonEt;
        float puIso = mu.pfIsolationR04().sumPUPt;
        mu_relIso_.push_back((chIso + std::max(0.0f, nhIso + phIso - 0.5f*puIso)) / mu.pt());
        if (mu.muonBestTrack().isNonnull()) {
            mu_dxy_.push_back(mu.muonBestTrack()->dxy(pv.position()));
            mu_dz_.push_back(mu.muonBestTrack()->dz(pv.position()));
        } else {
            mu_dxy_.push_back(0.0);
            mu_dz_.push_back(0.0);
        }
        mu_isLoose_.push_back(mu.isLooseMuon()   ? 1 : 0);
        mu_isMedium_.push_back(mu.isMediumMuon() ? 1 : 0);
        mu_isTight_.push_back(mu.isTightMuon(pv) ? 1 : 0);
        nMuons_++;
    }

    edm::Handle<pat::ElectronCollection> electrons;
    iEvent.getByToken(electronToken_, electrons);
    nElectrons_ = 0;
    if (electrons.isValid()) for (const pat::Electron& el : *electrons) {
        if (el.pt() < 10.0 || fabs(el.eta()) > 2.5) continue;
        el_pt_.push_back(el.pt());
        el_eta_.push_back(el.eta());
        el_phi_.push_back(el.phi());
        el_energy_.push_back(el.energy());
        el_mass_.push_back(el.mass());
        el_charge_.push_back(el.charge());
        float chIso = el.pfIsolationVariables().sumChargedHadronPt;
        float nhIso = el.pfIsolationVariables().sumNeutralHadronEt;
        float phIso = el.pfIsolationVariables().sumPhotonEt;
        float puIso = el.pfIsolationVariables().sumPUPt;
        el_relIso_.push_back((chIso + std::max(0.0f, nhIso + phIso - 0.5f*puIso)) / el.pt());
        if (el.gsfTrack().isNonnull()) {
            el_dxy_.push_back(el.gsfTrack()->dxy(pv.position()));
            el_dz_.push_back(el.gsfTrack()->dz(pv.position()));
        } else {
            el_dxy_.push_back(0.0);
            el_dz_.push_back(0.0);
        }
        nElectrons_++;
    }

    edm::Handle<pat::METCollection> mets;
    iEvent.getByToken(metToken_, mets);
    met_pt_ = met_phi_ = met_px_ = met_py_ = met_sumet_ = 0.0f;
    if (mets.isValid() && !mets->empty()) {
        met_pt_    = mets->front().pt();
        met_phi_   = mets->front().phi();
        met_px_    = mets->front().px();
        met_py_    = mets->front().py();
        met_sumet_ = mets->front().sumEt();
    }

    // ================================================================
    // LEADING-JET STUDY (MiniAOD/PF implementation)
    // ================================================================
    //
    // Common selection for EVERY variable (theta_J, lambda_J, epsilon_J,
    // nTracks, tau1, tau31) and for BOTH samples:
    //   * anti-kT R=0.4 jets from slimmedJets
    //   * pT > 50 GeV and |eta| < 2.4
    //   * leading jet = highest-pT jet passing both cuts
    //
    // Tau (signal) sample ONLY: the leading jet must have at least
    // MIN_TAUS_IN_JET reconstructed (slimmedTaus, decayModeFinding) taus
    // within DeltaR < TAU_JET_MAX_DR, plus the di-tau selection further
    // below. No generator-level information is used for the selection.
    // QCD: no tau selection at all, nothing beyond the leading jet.
    //
    // We use the already reconstructed AK4 PF jets in slimmedJets, so
    // theta_J is approximated with pat::Jet's PF energy-fraction accessors
    // (chargedHadronEnergyFraction() + neutralHadronEnergyFraction() +
    // HFHadronEnergyFraction()), which correctly count charged-hadron
    // energy as hadronic -- see the detailed note at the jet loop below.
    //
    // IMPORTANT: every variable is a property of the LEADING JET, not a
    // slimmed tau object.
    // ================================================================

    edm::Handle<std::vector<reco::GenParticle>> genParticles;
    iEvent.getByToken(genToken_, genParticles);
    if (!genParticles.isValid()) return;

    edm::Handle<pat::JetCollection> jets;
    iEvent.getByToken(jetToken_, jets);
    nJets_ = 0;
    if (!jets.isValid()) return;

    float bestLeadingPt = -1.0f;
    float bestEJ = 0.0f;
    float bestEHCAL = 0.0f;
    int bestNConstituents = 0;
    int bestNHCALConstituents = 0;
    int bestIndex = -1;
    // Pointer to the leading pat::Jet itself (not just its filtered-vector
    // index), needed below to re-cluster its PF constituents for Fig. 2.
    // Safe to keep for the rest of analyze(): it points into the vector
    // owned by the 'jets' Handle, which stays valid for the whole call.
    const pat::Jet* bestJetPtr = nullptr;
    bool anyJetPassesPt = false;   // for the cutflow only
    // Second-leading selected jet (j2), tracked by its index in the jet_* vectors.
    // Only used by the tau sample's di-tau selection.
    int secondIndex = -1;
    float secondLeadingPt = -1.0f;

    for (const pat::Jet& jet : *jets) {
        // Common leading-jet cuts: pT > 50 GeV and |eta| < 2.4.
        if (jet.pt() <= LEADING_JET_MIN_PT) continue;
        anyJetPassesPt = true;
        if (std::fabs(jet.eta()) >= LEADING_JET_MAX_ETA) continue;

        jet_pt_.push_back(jet.pt());
        jet_eta_.push_back(jet.eta());
        jet_phi_.push_back(jet.phi());
        jet_energy_.push_back(jet.energy());
        jet_mass_.push_back(jet.mass());

        float csv = jet.bDiscriminator("pfCombinedInclusiveSecondaryVertexV2BJetTags");
        if (csv < -0.5)
            csv = jet.bDiscriminator("combinedInclusiveSecondaryVertexV2BJetTags");
        if (csv < -0.5)
            csv = jet.bDiscriminator("combinedSecondaryVertexBJetTags");

        jet_csv_.push_back(csv);
        jet_hadFlav_.push_back(jet.hadronFlavour());
        jet_btagLoose_.push_back (csv > CSVv2_LOOSE_WP  ? 1 : 0);
        jet_btagMedium_.push_back(csv > CSVv2_MEDIUM_WP ? 1 : 0);
        jet_btagTight_.push_back (csv > CSVv2_TIGHT_WP  ? 1 : 0);
        ++nJets_;

        // Per-jet accumulators, set below from pat::Jet's energy
        // fraction accessors (see explanation in the block below).
        float EJ = 0.0f;
        float EHCAL = 0.0f;
        int nConst = 0;
        int nHCAL = 0;

        // ------------------------------------------------------------
        // theta_J = hadronic (HCAL-like) energy fraction of the jet.
        //
        // NOTE: We do NOT reconstruct this by summing daughter energies
        // weighted by PackedCandidate::hcalFraction(). That accessor is
        // only meaningful for neutral hadrons -- charged hadrons (the
        // dominant constituent of tau/QCD jets: charged pions, kaons,
        // protons) are stored with hcalFraction()==0 by convention in
        // MiniAOD, since their energy comes from the track, not a
        // calorimeter-tower split. Using it here would silently drop
        // most of the true hadronic energy and give a noisy, physically
        // wrong theta_J for exactly the samples where the paper's
        // variable is supposed to peak near 1.
        //
        // Instead, use pat::Jet's own PF energy-fraction accessors,
        // which correctly classify charged-hadron energy as hadronic:
        //   hadronic (HCAL-like) = chargedHadronEnergyFraction()
        //                        + neutralHadronEnergyFraction()
        //                        + HFHadronEnergyFraction()
        //   EM (ECAL-like)       = neutralEmEnergyFraction()   (photons)
        //                        + chargedEmEnergyFraction()   (electrons)
        //                        + HFEMEnergyFraction()
        // The two sides sum to ~1 (muonEnergyFraction() is the small
        // remainder, physically MIP-like and not assigned to either).
        // ------------------------------------------------------------
        const float chHadFrac = jet.chargedHadronEnergyFraction();
        const float neHadFrac = jet.neutralHadronEnergyFraction();
        const float hfHadFrac = jet.HFHadronEnergyFraction();

        EJ = static_cast<float>(jet.energy());
        EHCAL = EJ * (chHadFrac + neHadFrac + hfHadFrac);
        nConst = static_cast<int>(jet.numberOfDaughters());
        nHCAL = -1; // not meaningful with the fraction-based approach

        // The leading jet (j1) is the highest-pT jet passing pT > 50 GeV and |eta| < 2.4;
        // the second-leading jet (j2) is the next one.
        if (jet.pt() > bestLeadingPt) {
            // the previous leading jet (if any) becomes the second-leading jet
            secondLeadingPt = bestLeadingPt;
            secondIndex = bestIndex;
            bestLeadingPt = jet.pt();
            bestEJ = EJ;
            bestEHCAL = EHCAL;
            bestNConstituents = nConst;
            bestNHCALConstituents = nHCAL;
            bestIndex = nJets_ - 1;
            bestJetPtr = &jet;
        } else if (jet.pt() > secondLeadingPt) {
            secondLeadingPt = jet.pt();
            secondIndex = nJets_ - 1;
        }
    }

    // No qualifying leading jet -> event dropped.
    if (anyJetPassesPt) h_cutflow_->Fill(CF_JET_PT);
    if (bestIndex < 0) return;
    h_cutflow_->Fill(CF_LEADING_JET);

    leadingJetIndex_ = bestIndex;
    leadingJetNConstituents_ = bestNConstituents;
    leadingJetNHCALConstituents_ = bestNHCALConstituents;

    leadingJetPt_ = jet_pt_[bestIndex];
    leadingJetEta_ = jet_eta_[bestIndex];
    leadingJetPhi_ = jet_phi_[bestIndex];
    leadingJetE_ = jet_energy_[bestIndex];

    leadingJetEJ_ = bestEJ;
    leadingJetEHCAL_ = bestEHCAL;

    // theta_J = E_HCAL / E_J.
    // IMPORTANT: these are validity checks for theta_J specifically (it's
    // a ratio involving E_HCAL). lambda_J, epsilon_J, nTracks and tau are
    // independent quantities, so a bad/undefined theta_J must NOT skip the
    // rest of the event. Track validity with a flag and only gate the
    // h_logThetaJ_ fill on it, instead of `return`-ing out of analyze()
    // early. The branch keeps its sentinel (-99) when theta_J is undefined.
    bool validThetaJ = true;

    if (leadingJetEJ_ <= 0.0f) {
        validThetaJ = false;
    } else {
        leadingJetThetaJ_ = leadingJetEHCAL_ / leadingJetEJ_;

        // Physical fraction should be in [0,1]. Numerical safety only.
        if (leadingJetThetaJ_ < 0.0f) {
            validThetaJ = false;
        } else {
            if (leadingJetThetaJ_ > 1.0f) leadingJetThetaJ_ = 1.0f;

            // log10(theta_J) is undefined at theta_J = 0, which happens
            // for purely-ECAL jets (essentially zero HCAL deposit). Drop
            // just the theta_J entry rather than flooring it into the
            // leftmost bin -- flooring artificially piles every such jet
            // onto a single bin edge, which isn't the physical
            // distribution.
            if (leadingJetThetaJ_ <= 0.0f) {
                validThetaJ = false;
            } else {
                leadingJetLogThetaJ_ = std::log10(leadingJetThetaJ_);
            }
        }
    }

    // ---------------------------------------------------------------
    // TAU (SIGNAL) SAMPLE ONLY. QCD skips this whole block: no tau
    // matching, no di-tau requirement, no tau cutflow bins. For QCD,
    // nTausInJet, secondJetIndex and diTauPair keep their -1 sentinels.
    //
    //   step 1: at least MIN_TAUS_IN_JET RECONSTRUCTED tau (slimmedTaus,
    //           passing the baseline "decayModeFinding" discriminator)
    //           within DeltaR < TAU_JET_MAX_DR of the leading jet j1.
    //           No gen-particle information is used.
    //   step 2: at least 2 selected jets. j1 = leading jet, j2 =
    //           second-leading jet (same pT > 50, |eta| < 2.4 cuts).
    //   step 3: two DIFFERENT reco taus, one within dR < TAU_JET_MAX_DR
    //           of j1 and the other within dR < TAU_JET_MAX_DR of j2
    //           (either pairing).
    // ---------------------------------------------------------------
    if (isTauSample_) {
        edm::Handle<pat::TauCollection> taus;
        iEvent.getByToken(tauToken_, taus);
        std::vector<std::pair<double, double> > tauEtaPhi;   // every tau passing decayModeFinding
        if (taus.isValid()) {
            int nTaus = 0;
            for (const pat::Tau& tau : *taus) {
                if (tau.tauID("decayModeFinding") < 0.5) continue;
                tauEtaPhi.push_back(std::make_pair((double)tau.eta(), (double)tau.phi()));

                const double dR = reco::deltaR(
                    leadingJetEta_, leadingJetPhi_, tau.eta(), tau.phi());

                if (dR < TAU_JET_MAX_DR) ++nTaus;
            }
            nTausInJet_ = nTaus;
        }

        // nTausInJet_ stays -1 if the tau collection is missing, which fails this cut.
        if (nTausInJet_ < MIN_TAUS_IN_JET) return;
        h_cutflow_->Fill(CF_TAU_MATCHED);

        secondJetIndex_ = secondIndex;
        if (secondIndex < 0) return;
        h_cutflow_->Fill(CF_TWO_JETS);

        const bool tauPairOK = tauPairMatchesJets(tauEtaPhi,
                                   leadingJetEta_, leadingJetPhi_,
                                   jet_eta_[secondIndex], jet_phi_[secondIndex],
                                   TAU_JET_MAX_DR);
        diTauPair_ = tauPairOK ? 1 : 0;
        if (!tauPairOK) return;
        h_cutflow_->Fill(CF_TAU_PAIR);
    }

    // Fill the Fig. 1 (left) histogram: one entry per selected event
    // where theta_J was well-defined, using the leading jet that passed
    // the common jet cuts and (for the tau sample) the reco-tau
    // requirements above.
    if (validThetaJ) h_logThetaJ_->Fill(leadingJetLogThetaJ_);

    // ------------------------------------------------------------------
    // Fig. 2: lambda_J (left) and epsilon_J (right), Eq. (3), for the
    // same selected leading jet. Both are built by re-clustering the
    // jet's own PF constituents with the kT algorithm into a fixed
    // number of exclusive subjets (nt=5 for lambda_J, nf=3 for
    // epsilon_J), following Ref.[16,17] as cited in the paper.
    // ------------------------------------------------------------------
    if (bestJetPtr != nullptr && leadingJetPt_ > 0.0f) {
        // --- lambda_J = log(1 - pT_0 / pT_J), using nt=5 subjets ---
        std::vector<fastjet::PseudoJet> subjetsNt;
        if (getExclusiveSubjets(*bestJetPtr, NT_SUBJETS, subjetsNt)) {
            const double pT0 = subjetsNt[0].pt();      // hardest subjet
            const double pTJ = leadingJetPt_;
            const double ratio = pT0 / pTJ;
            // ratio -> 1 for a single pencil-like prong, per Eq. (4);
            // log(1-ratio) is only defined for ratio < 1.
            if (ratio < 1.0) {
                leadingJetLambdaJ_ = static_cast<float>(std::log(1.0 - ratio));
                h_lambdaJ_->Fill(leadingJetLambdaJ_);
            }
        }

        // --- epsilon_J = (1/E_J^2) * sum_{nf>=i>j} E_i*E_j, nf=3 subjets ---
        std::vector<fastjet::PseudoJet> subjetsNf;
        if (getExclusiveSubjets(*bestJetPtr, NF_SUBJETS, subjetsNf)) {
            const double EJ = bestJetPtr->energy();
            double sumEiEj = 0.0;
            for (size_t i = 0; i < subjetsNf.size(); ++i)
                for (size_t j = 0; j < i; ++j)
                    sumEiEj += subjetsNf[i].E() * subjetsNf[j].E();
            if (EJ > 0.0) {
                leadingJetEpsilonJ_ = static_cast<float>(sumEiEj / (EJ * EJ));
                h_epsilonJ_->Fill(leadingJetEpsilonJ_);
            }
        }
    }

    // ------------------------------------------------------------------
    // Additional jet-shape / N-subjettiness variables (nTracks, tau1,
    // tau31, AK8 versions), computed from the SAME leading AK4 jet
    // (bestJetPtr) already selected above for theta_J/lambda_J/epsilon_J.
    // These only set the tree branches -- they do not gate tree_->Fill()
    // or affect any of the existing histograms/branches.
    // ------------------------------------------------------------------
    if (bestJetPtr != nullptr) {
        // --- nTracks: charged PF candidates from the PV within dR<0.4 of
        //     the leading jet axis, pt>2 GeV ---
        edm::Handle<pat::PackedCandidateCollection> packedCandidates;
        iEvent.getByToken(packedCandidateToken_, packedCandidates);
        if (packedCandidates.isValid()) {
            int nTracks = 0;
            for (const auto& cand : *packedCandidates) {
                if (cand.charge() == 0 || cand.pt() <= 2.0 || cand.fromPV() < 2) continue;
                float dR = reco::deltaR(cand.eta(), cand.phi(), leadingJetEta_, leadingJetPhi_);
                if (dR < 0.4) ++nTracks;
            }
            nTracks_ = nTracks;
            h_nTracks_->Fill(nTracks);
        }

        // --- tau1 / tau31 (AK4, custom N-subjettiness) ---
        double tau1 = calculateNSubjettiness(*bestJetPtr, 1, 0.4);
        double tau3 = calculateNSubjettiness(*bestJetPtr, 3, 0.4);
        if (tau1 > 0.0 && tau1 <= 1.0) {
            tau1_ = (float)tau1;
            double logTau1 = std::log(tau1);
            if (logTau1 >= -3.0 && logTau1 <= 0.0) h_logTau1_->Fill(logTau1);
        }
        if (tau1 > 0.0 && tau3 >= 0.0) {
            tau31_ = (float)(tau3 / tau1);
            if (tau31_ >= 0.0 && tau31_ <= 0.9) h_tau31_->Fill(tau31_);
        }
    }

    // --- AK8: read stored NjettinessAK8:tau1/tau3 userFloats directly.
    //     NOTE: this uses its OWN leading AK8 jet (pt>170, |eta|<2.4); it is
    //     not tied to the leading AK4 jet above. The BDT uses the AK4
    //     tau1/tau31 branches, not these. ---
    {
        edm::Handle<pat::JetCollection> jetsAK8;
        iEvent.getByToken(jetAK8Token_, jetsAK8);

        const pat::Jet* leadingAK8 = nullptr;
        float leadingPtAK8 = -1.0f;
        if (jetsAK8.isValid()) {
            for (const auto& jet : *jetsAK8) {
                if (jet.pt() < 170.0 || std::fabs(jet.eta()) > 2.4) continue;
                if (jet.pt() > leadingPtAK8) { leadingPtAK8 = jet.pt(); leadingAK8 = &jet; }
            }
        }

        if (leadingAK8
            && leadingAK8->hasUserFloat("NjettinessAK8:tau1")
            && leadingAK8->hasUserFloat("NjettinessAK8:tau3")) {
            const double tau1AK8 = leadingAK8->userFloat("NjettinessAK8:tau1");
            const double tau3AK8 = leadingAK8->userFloat("NjettinessAK8:tau3");
            if (tau1AK8 > 0.0 && tau1AK8 <= 1.0) {
                tau1AK8_ = (float)tau1AK8;
                double logTau1AK8 = std::log(tau1AK8);
                if (logTau1AK8 >= -3.0 && logTau1AK8 <= 0.0) h_logTau1AK8_->Fill(logTau1AK8);
            }
            if (tau1AK8 > 0.0 && tau3AK8 >= 0.0) {
                tau31AK8_ = (float)(tau3AK8 / tau1AK8);
                if (tau31AK8_ >= 0.0 && tau31AK8_ <= 0.9) h_tau31AK8_->Fill(tau31AK8_);
            }
        }
    }

    nGenLep_ = 0;
    nGenPhotons_ = 0;
    for (const reco::GenParticle& gp : *genParticles) {
        int absPdg = abs(gp.pdgId());

        if (absPdg == 11 || absPdg == 13 || absPdg == 12 || absPdg == 14) {
            if (gp.status() != 1 && gp.status() != 23) continue;
            gen_pt_.push_back(gp.pt());
            gen_eta_.push_back(gp.eta());
            gen_phi_.push_back(gp.phi());
            gen_mass_.push_back(gp.mass());
            gen_pdgId_.push_back(gp.pdgId());
            gen_status_.push_back(gp.status());
            int motherPdg = 0;
            if (gp.numberOfMothers() > 0) motherPdg = gp.mother(0)->pdgId();
            gen_motherPdgId_.push_back(motherPdg);
            nGenLep_++;
            continue;
        }

        if (gp.pdgId() == 22) {
           // these are gen photons. 
            if (gp.pt() < 5.0) continue;
            gen_pho_pt_.push_back(gp.pt());
            gen_pho_eta_.push_back(gp.eta());
            gen_pho_phi_.push_back(gp.phi());
            gen_pho_energy_.push_back(gp.energy());
            gen_pho_status_.push_back(gp.status());
            gen_pho_isPrompt_.push_back(gp.isPromptFinalState() ? 1 : 0);
            int motherPdg = 0;
            if (gp.numberOfMothers() > 0) motherPdg = gp.mother(0)->pdgId();
            gen_pho_motherPdgId_.push_back(motherPdg);
            nGenPhotons_++;
        }
    }

    // ---- these are reco photons 
edm::Handle<pat::PhotonCollection> photons;
iEvent.getByToken(photonToken_, photons);
nPhotons_ = 0;
if (photons.isValid()) for (const pat::Photon& pho : *photons) {
    if (pho.pt() < 15.0)          continue;   
    if (fabs(pho.eta()) > 2.5)    continue;   
    // Exclude ECAL gap: 1.4442 < |eta| < 1.566 
    if (fabs(pho.eta()) > 1.4442 && fabs(pho.eta()) < 1.566) continue;
    pho_pt_.push_back(pho.pt());
    pho_eta_.push_back(pho.eta());
    pho_phi_.push_back(pho.phi());
    pho_energy_.push_back(pho.energy());
  
    pho_sigmaIetaIeta_.push_back(pho.sigmaIetaIeta());

    pho_hOverE_.push_back(pho.hadTowOverEm());
  
    float chIso = pho.chargedHadronIso();
    float nhIso = pho.neutralHadronIso();
    float phIso = pho.photonIso();
    float relIso = (chIso + nhIso + phIso) / pho.pt();
    pho_relIso_.push_back(relIso);
 
    int passLoose = 0;
    if (pho.isPhotonIDAvailable("cutBasedPhotonID-Spring15-25ns-V1-standalone-loose"))
        passLoose = pho.photonID("cutBasedPhotonID-Spring15-25ns-V1-standalone-loose") > 0.5 ? 1 : 0;
    pho_passLooseId_.push_back(passLoose);

    // Gen-match this reco photon to the nearest gen photon (dR < 0.1) to understand whether its prompt photon or radiated photon
    int matchedPrompt = -1;
    int matchedStatus = -1;
    float bestDR = 0.1;
    for (size_t k = 0; k < gen_pho_eta_.size(); ++k) {
        float dR = reco::deltaR(pho.eta(), pho.phi(), gen_pho_eta_[k], gen_pho_phi_[k]);
        if (dR < bestDR) {
            bestDR = dR;
            matchedPrompt = gen_pho_isPrompt_[k];
            matchedStatus = gen_pho_status_[k];
        }
    }
    pho_isPromptMatched_.push_back(matchedPrompt);
    pho_matchedGenStatus_.push_back(matchedStatus);

    nPhotons_++;
}

    // ---- GENERATOR WEIGHT ----
    edm::Handle<GenEventInfoProduct> genInfo;
    iEvent.getByToken(genInfoToken_, genInfo);
    genWeight_ = genInfo.isValid() ? (float)genInfo->weight() : 1.0f;

    // ---- MET FILTERS ----
    flag_goodVertices_ = flag_HBHENoiseFilter_ = flag_HBHENoiseIsoFilter_ =
        flag_EcalDeadCellTriggerPrimitiveFilter_ = flag_eeBadScFilter_ =
        flag_globalTightHalo_ = -1;
    edm::Handle<edm::TriggerResults> metFilterResults;
    iEvent.getByToken(metFilterToken_, metFilterResults);
    if (metFilterResults.isValid()) {
        const edm::TriggerNames& filterNames = iEvent.triggerNames(*metFilterResults);
        for (unsigned int i = 0; i < filterNames.size(); ++i) {
            const std::string& name = filterNames.triggerName(i);
            int decision = metFilterResults->accept(i) ? 1 : 0;
            if      (name == "Flag_goodVertices")                       flag_goodVertices_ = decision;
            else if (name == "Flag_HBHENoiseFilter")                    flag_HBHENoiseFilter_ = decision;
            else if (name == "Flag_HBHENoiseIsoFilter")                 flag_HBHENoiseIsoFilter_ = decision;
            else if (name == "Flag_EcalDeadCellTriggerPrimitiveFilter") flag_EcalDeadCellTriggerPrimitiveFilter_ = decision;
            else if (name == "Flag_eeBadScFilter")                      flag_eeBadScFilter_ = decision;
            else if (name == "Flag_globalTightHalo2016Filter")          flag_globalTightHalo_ = decision;
        }
    }

    
    // The tree is filled once per event, after the common leading-jet
    // selection and (tau sample only) the reco-tau, two-jet and di-tau pair
    // requirements above. No lepton / photon skim is applied.
    tree_->Fill();
    h_cutflow_->Fill(CF_FILLED);
}

// ----------------------------------------------------------------------
// Re-cluster a jet's own PF constituents (the daughters MiniAOD already
// stores on the pat::Jet) with the kT algorithm and read off exactly
// nSubjets *exclusive* subjets -- i.e. the state of the kT clustering
// history at the point where merging stops with precisely nSubjets
// pseudojets left. This is the standard "subjet" construction used for
// jet substructure and is exactly what pT_i, E_i in Eq. (3)-(4) refer to.
//
// R is set large (1.5) purely so that a single kT clustering run
// (radius doesn't matter much here since we only ever look at the
// *exclusive* history, not which inclusive jets come out at the end)
// merges all of the jet's constituents together; exclusive_subjets()
// then unwinds that single merge tree back to the requested multiplicity.
// ----------------------------------------------------------------------
bool WZAnalyzer::getExclusiveSubjets(const pat::Jet& jet, unsigned int nSubjets,
                                      std::vector<fastjet::PseudoJet>& subjets) {
    std::vector<fastjet::PseudoJet> constituents;
    constituents.reserve(jet.numberOfDaughters());
    for (unsigned int i = 0; i < jet.numberOfDaughters(); ++i) {
        const reco::Candidate* dau = jet.daughter(i);
        if (!dau) continue;
        constituents.emplace_back(dau->px(), dau->py(), dau->pz(), dau->energy());
    }

    // Can't form more exclusive subjets than there are input particles.
    if (constituents.size() < nSubjets) return false;

    fastjet::JetDefinition jetDef(fastjet::kt_algorithm, 1.5);
    fastjet::ClusterSequence clusterSeq(constituents, jetDef);

    std::vector<fastjet::PseudoJet> inclusiveJets = clusterSeq.inclusive_jets(0.0);
    if (inclusiveJets.empty()) return false;

    // With R=1.5 (much larger than the AK4 jet the constituents came
    // from) all constituents should land in one inclusive pseudojet;
    // take the hardest one defensively in case of edge cases.
    std::sort(inclusiveJets.begin(), inclusiveJets.end(),
        [](const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
            return a.pt() > b.pt();
        });

    subjets = clusterSeq.exclusive_subjets(inclusiveJets.front(), static_cast<int>(nSubjets));
    if (subjets.size() < nSubjets) return false;

    // Order subjets by descending pT so subjets[0] is "the 0-th subjet"
    // (the hardest) as used in Eq. (3).
    std::sort(subjets.begin(), subjets.end(),
        [](const fastjet::PseudoJet& a, const fastjet::PseudoJet& b) {
            return a.pt() > b.pt();
        });

    return true;
}

void WZAnalyzer::endJob() {
    // Print the cutflow table for THIS job (one input file). For the total over all
    // files, hadd the outputs and run make_cutflow_table.py.
    // The tau, two-jet and tau-pair steps exist only for the tau sample, so they
    // are not printed for QCD. "% of prev" is relative to the previous printed step.
    std::cout << "\n[WZAnalyzer] cutflow ("
              << (isTauSample_ ? "tau / signal sample" : "QCD / background sample") << ")\n";

    char line[256];
    std::snprintf(line, sizeof(line), "  %-36s %12s %10s %10s\n",
                  "step", "events", "% of prev", "% of all");
    std::cout << line;

    const double total = h_cutflow_->GetBinContent(CF_ALL);
    double prev = total;
    for (int b = 1; b <= CF_NBINS; ++b) {
        const bool tauStep = (b == CF_TAU_MATCHED || b == CF_TWO_JETS || b == CF_TAU_PAIR);
        if (tauStep && !isTauSample_) continue;   // QCD: no tau steps at all

        const double n = h_cutflow_->GetBinContent(b);
        const double ofPrev = (b == CF_ALL) ? 100.0 : (prev > 0.0 ? 100.0 * n / prev : 0.0);
        const double ofAll  = total > 0.0 ? 100.0 * n / total : 0.0;
        std::snprintf(line, sizeof(line), "  %-36s %12.0f %9.2f%% %9.2f%%\n",
                      h_cutflow_->GetXaxis()->GetBinLabel(b), n, ofPrev, ofAll);
        std::cout << line;
        prev = n;
    }
    std::cout << std::endl;
}
DEFINE_FWK_MODULE(WZAnalyzer);


