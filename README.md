# Recognition Without Prescription

Code and analysis scripts for "Recognition Without Prescription: Defamiliarizing Relationship Advice Through LLM Divergence"

## Overview

This repository contains the code used to analyze how LLM-generated advice diverges from community-endorsed advice on r/relationship_advice. The analysis demonstrates that LLMs recognize relationship problems using diagnostic vocabulary but systematically fail to prescribe the solutions the community endorses—a pattern we call "recognition without prescription."

## Repository Structure

```
recognition-without-prescription/
├── data/                               # Data files (Git LFS)
│   ├── llm_advice.parquet              # LLM-generated advice responses
│   ├── stratified_sample.parquet       # Stratified sample of Reddit posts
│   ├── advice_metrics.parquet          # Computed linguistic metrics
│   ├── permission_metrics.parquet      # Permission structure analysis
│   ├── multi_model_assignments.parquet # Topic classifications
│   ├── pairwise_validation.csv         # Validation sample pairs
│   └── persona_comparison.csv          # Persona prompting results
│
├── scripts/
│   ├── data/                           # Data preparation
│   │   ├── clean_data.py               # Clean raw Reddit data
│   │   ├── stratified_sample.py        # Create stratified sample
│   │   └── generate_llm_advice.py      # Generate LLM advice responses
│   │
│   ├── topic_modeling/                 # Topic classification
│   │   └── multi_model_assignment.py   # Multi-model topic assignment
│   │
│   └── analysis/                               # Analysis scripts
│       ├── compute_advice_metrics.py           # Core linguistic metrics
│       ├── defamiliarization_analysis.py       # Divergence analysis
│       ├── permission_granting_analysis.py     # Permission spectrum
│       ├── persona_prompting_check.py          # Robustness check
│       ├── prepare_validation_sample.py        # Validation sample
│       └── categorization_advice_analysis.py   # Categorization-advice link
│
├── notebooks/                                  # Figure generation
│   ├── paper_figures.ipynb                     # Main paper figures
│   ├── fig_consensus_divergence.ipynb          # Figure: consensus divergence
│   ├── fig_therapy_simple.ipynb                # Figure: therapy flattening
│   ├── advice_metrics_analysis.ipynb           # Metrics analysis
│   └── multi_model_assignment_analysis.ipynb   # Topic analysis
│
├── config/
│   └── config.py                # OpenRouter model configuration
│
└── README.md
```

## Pipeline

### 1. Data Preparation

```bash
# Clean raw Reddit data
python scripts/data/clean_data.py

# Create stratified sample
python scripts/data/stratified_sample.py

# Generate LLM advice (requires OpenRouter API key)
python scripts/data/generate_llm_advice.py
```

### 2. Topic Classification

```bash
# Assign topics using multiple models
python scripts/topic_modeling/multi_model_assignment.py
```

### 3. Analysis

```bash
# Compute linguistic metrics (leave ratio, certainty, hedging, therapy, etc.)
python scripts/analysis/compute_advice_metrics.py

# Run defamiliarization analysis
python scripts/analysis/defamiliarization_analysis.py

# Analyze permission structure
python scripts/analysis/permission_granting_analysis.py

# Robustness check: persona prompting
python scripts/analysis/persona_prompting_check.py
```

### 4. Figures

Jupyter notebooks in `notebooks/` generate paper figures.

## Requirements

```
pandas
numpy
spacy
nltk
scipy
matplotlib
seaborn
openai  # For OpenRouter API calls
```

## Data

Data files are stored using [Git LFS](https://git-lfs.github.com/) (Large File Storage). To clone with data:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone repository (LFS files download automatically)
git clone https://github.com/tomvannuenen/recognition-without-prescription.git
```

### Data Files

| File | Description |
|------|-------------|
| `llm_advice.parquet` | LLM-generated advice responses from 4 models |
| `stratified_sample.parquet` | Stratified sample of Reddit posts with human comments |
| `advice_metrics.parquet` | Computed linguistic metrics (leave ratio, certainty, hedging, etc.) |
| `permission_metrics.parquet` | Permission structure analysis results |
| `multi_model_assignments.parquet` | Topic classifications from multi-model assignment |
| `pairwise_validation.csv` | Sample pairs for human validation |
| `persona_comparison.csv` | Persona prompting robustness check results |

### Source Data

The analysis uses:
- Reddit posts from r/relationship_advice (Oct-Dec 2025)
- Top-voted human comments per post
- LLM-generated advice from 4 models (Gemini 2.5 Flash Lite, DeepSeek v3.2, Ministral 8B, GPT-4.1-nano)
