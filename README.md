# A Bi-Directional Exploration of Anxiety and Attentional States in Pilots  
**Causal Analysis and Predictive Modelling using Multimodal Physiological Data**

## Project Overview

This repository contains the full codebase for the final-year research project conducted under the  
**B.Sc. (Honours) in Data Science**,  
**Department of Statistics, Faculty of Science, University of Colombo**.

The project investigates the **bi-directional relationship between pilot attentional states and anxiety** using multimodal physiological data, and develops machine learning models to **predict attentional states** from physiological signals.

The work is motivated by aviation safety, human factors research, and the role of physiological monitoring in detecting cognitive and affective states in high-stakes environments.

---

## Research Objectives

This project addresses two primary research questions:

1. **Causal Analysis**  
   Do changes in pilot attentional states (Channelized Attention, Diverted Attention, Startle/Surprise) cause measurable changes in:
   - Somatic anxiety (e.g., heart rate, skin conductance, respiration)?
   - Cognitive anxiety (e.g., EEG-derived indicators)?

2. **Predictive Modelling**  
   Can attentional states be accurately predicted from multimodal physiological data using machine learning models trained on controlled benchmark data and evaluated on full flight simulations?

---

## Dataset

**Source:**  
NASA Open Data Portal — *Flight Crew Physiological Data for Crew State Monitoring*

**Data Description:**
- Physiological signals:
  - EEG (multiple channels)
  - ECG
  - Galvanic Skin Response (GSR)
  - Respiration
- Subjects:
  - 17 pilots (9 pilot–copilot crews)
- Experimental structure:
  - **Benchmark sessions**: Controlled tasks  
    - Channelized Attention (CA)  
    - Diverted Attention (DA)  
    - Startle/Surprise (SS)
  - **LOFT sessions**: Full-flight simulations (take-off, flight, landing)

**Important Note on Data Handling:**  
Raw and processed physiological data are **not included in this repository** due to data size and ethical considerations. The repository contains only code and documentation required to reproduce all analyses once the data are obtained from NASA.

---

## Methodological Overview

The project follows a **strict, reproducible pipeline** designed to avoid common pitfalls such as subject leakage, temporal dependence violations, and improper validation.

### 1. Data Ingestion & Standardization
- Read raw CSV files from NASA dataset
- Standardize column names and labels
- Convert ADC counts to physical units
- Store cleaned session-level files locally

### 2. Signal Preprocessing
- Filtering (EEG, ECG, GSR, respiration)
- Artifact detection and quality control
- R-peak detection (ECG)
- Decomposition of tonic/phasic GSR components

### 3. Windowing
- Sliding windows with overlap
- Guard bands to remove transition windows
- Window-level labeling based on majority class

### 4. Feature Extraction
- EEG band power features (delta, theta, alpha, beta)
- Heart rate and heart rate variability metrics
- GSR phasic and tonic features
- Respiration rate and variability
- Multi-scale and lagged physiological features

### 5. Anxiety Indices
Two subject-normalized indices are constructed:
- **Somatic Anxiety Index (SAI)**  
  Derived from ECG, GSR, and respiration features
- **Cognitive Anxiety Index (CAI)**  
  Derived from frontal EEG band power relationships

### 6. Causal Analysis
- Event-study framework using mixed-effects models
- Subject-level random effects
- Pre-trend checks and robustness analyses
- Focus on temporal dynamics around attentional event onsets

### 7. Predictive Modelling
- Models trained **only on benchmark data**
- Leave-one-crew-out cross-validation
- Classical models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Deep learning models:
  - CNN-based architectures
  - Multimodal encoders
  - Optional subject-aware embeddings
- Final evaluation performed **once** on LOFT data

---

## Repository Structure
project-root/
├── raw/ # Raw NASA data (local only, not tracked)
├── processed/ # Cleaned per-session files (local)
├── preprocessed/ # Filtered signals (local)
├── windows/ # Windowed data (local)
├── features/ # Extracted features (local)
├── models/ # Saved model checkpoints (local)
├── results/ # Plots and evaluation tables (local)
├── notebooks/ # Optional exploratory notebooks
├── src/ # Core source code
│ ├── ingest.py
│ ├── qc.py
│ ├── preprocess.py
│ ├── windowing.py
│ ├── features.py
│ ├── causal_analysis.py
│ ├── train_classical.py
│ ├── train_dl.py
│ ├── evaluate.py
│ └── utils.py
├── .github/ # GitHub workflows (CI)
├── environment.yml
├── requirements.txt
├── Dockerfile
└── README.md


All data-related directories are excluded via `.gitignore`.

---

## Environment Setup

The project uses **Python 3.10** and Conda for environment management.

Create the environment:

```bash
conda env create -f environment.yml
conda activate pilot-physio

Alternatively, install dependencies via:
pip install -r requirements.txt

Reproducibility & Best Practices
Subject-level normalization is computed only using benchmark data
LOFT data are reserved strictly for final evaluation
Cross-validation is grouped by crew to avoid leakage
Random seeds are fixed for reproducibility
All preprocessing parameters are logged and saved

Ethical Considerations
The dataset is de-identified but treated as sensitive
Raw physiological signals are not shared publicly
The repository provides code and instructions only
The study is conducted for academic research purposes

Author
K. J. A. Jayawardana
B.Sc. (Honours) in Data Science
University of Colombo

Supervisor
Dr. D. S. Wickramarachchi
Department of Statistics, University of Colombo

License
This repository is intended for academic and research use.
Please cite appropriately if any part of this work is referenced.


