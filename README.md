# DNN-architecture-for-particle-identification
# Identification of Tau Particles using Machine Learning and Deep Learning

This project focuses on the identification of hadronically decaying tau leptons (\(\tau_h\)) at the Large Hadron Collider (LHC), with particular emphasis on distinguishing genuine tau decays from QCD jets, electrons and muons.

The study explores classical machine learning techniques and a DeepTau-inspired neural network architecture using CMS Open Data.

## Project Overview

Hadronically decaying tau leptons are important in several particle physics analyses. However, their identification is challenging because their detector signatures can resemble those of quark- and gluon-initiated jets.

The project was carried out in two main stages:

1. **Classical Machine Learning:** Discrimination between hadronic tau candidates and QCD jets using jet-substructure observables.
2. **Deep Learning:** Development of a multi-branch neural network inspired by the CMS DeepTau architecture to classify four categories: hadronic tau decays, electrons, muons and QCD jets.

## Methodology

### 1. Event Processing

- Processed CMS MiniAOD datasets using the CMS Software (CMSSW).
- Adapted the WZAnalyzer to extract relevant reconstructed and particle-level information.
- Stored the extracted observables in ROOT files.
- Used separate CMSSW environments for the different datasets.

### 2. Jet-Substructure Analysis

Six jet-substructure observables were investigated:

- Hadronic energy fraction (\(\theta_J\))
- Number of charged tracks (\(N_{\mathrm{tracks}}\))
- Leading-subjet variable (\(\lambda_J\))
- Energy-sharing variable (\(\epsilon_J\))
- N-subjettiness (\(\tau_1\))
- N-subjettiness ratio (\(\tau_{31}\))

These observables were used to train and compare three classical machine learning classifiers:

- XGBoost
- Random Forest
- Gradient Boosting

### 3. DeepTau-Inspired Neural Network

A multi-branch neural network was developed using three input branches:

- **High-Level Features:** Reconstructed kinematic, isolation, impact parameter, track and multiplicity information.
- **Inner Grid:** An 11 × 11 particle-level grid with a cell size of 0.02 × 0.02.
- **Outer Grid:** A 21 × 21 particle-level grid with a cell size of 0.05 × 0.05.

The network combines fully connected layers and convolutional layers to learn information from both reconstructed observables and the spatial distribution of surrounding particles.

Two input configurations were investigated:
- 41 high-level observables.
- 47 inputs, consisting of 41 high-level observables and 6 jet-substructure observables.

The final network performs four-class classification using a Softmax output layer.

### 4. Training and Evaluation

- Applied feature transformations and standardisation.
- Used a 70:15:15 training, validation and testing split.
- Used class-weighted cross-entropy loss to address class imbalance.
- Trained the neural network using the Nadam optimiser.
- Evaluated performance using ROC curves, AUC, confusion matrices, tau-identification efficiency and background misidentification rates.

## Results

### Classical Machine Learning

| Classifier | Test AUC |
|---|---:|
| Initial XGBoost BDT | 0.825 |
| Improved XGBoost BDT | **0.895** |
| Random Forest | 0.894 |
| Gradient Boosting | 0.894 |

### DeepTau-Inspired Neural Network

| Configuration | Tau vs QCD AUC | Four-class Accuracy |
|---|---:|---:|
| 41 inputs | 0.8355 | 0.810 |
| 47 inputs | **0.8361** | **0.823** |

The addition of six jet-substructure observables improved electron and muon rejection, while QCD jet rejection remained nearly unchanged.

The neural network achieved excellent separation between tau candidates and electrons/muons. However, distinguishing hadronic tau decays from QCD jets remained the main challenge.

*Note: The classical BDT and neural network results correspond to different classification tasks and test samples, so their AUC values should not be interpreted as a direct comparison.*

## Technologies Used

- Python
- C++
- ROOT
- CMS Software (CMSSW)
- Docker
- XGBoost
- Scikit-learn
- TensorFlow / Keras
- NumPy
- Matplotlib

## Future Scope

Potential extensions of this work include:

- Exploring Transformer-based architectures for particle-level information.
- Studying the effect of different particle-grid configurations.
- Performing ablation studies of the network branches.
- Optimising network architecture and training hyperparameters.
- Evaluating performance across different transverse momentum and pseudorapidity ranges.
- Testing on independent datasets and comparing with established CMS tau-identification methods.

## References

1. CMS Collaboration, *Identification of hadronic tau lepton decays using a deep neural network*, Journal of Instrumentation, 17 (2022), P07023. [DOI](https://doi.org/10.1088/1748-0221/17/07/P07023).
2. CMS Open Data: https://opendata.cern.ch/
3. CMS Software (CMSSW): https://github.com/cms-sw/cmssw

---

**Project:** Identification of Tau Particles  
**Institution:** Department of Physics, IIT Delhi  
**Course:** PYD411 (2026–2027, Semester I)
