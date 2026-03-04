# Recognition Without Prescription

Code and analysis scripts for "Recognition Without Prescription: Defamiliarizing Relationship Advice Through LLM Divergence"

## Overview

This repository contains the code used to analyze how LLM-generated advice diverges from community-endorsed advice on r/relationship_advice. The analysis demonstrates that LLMs recognize relationship problems using diagnostic vocabulary but systematically fail to prescribe the solutions the community endorses—a pattern we call "recognition without prescription."

## Repository Structure

```
repo-release/
├── scripts/
│   ├── data/                    # Data preparation
│   │   ├── clean_data.py        # Clean raw Reddit data
│   │   ├── stratified_sample.py # Create stratified sample
│   │   └── generate_llm_advice.py # Generate LLM advice responses
│   │
│   ├── topic_modeling/          # Topic classification
│   │   └── multi_model_assignment.py # Multi-model topic assignment
│   │
│   └── analysis/                # Analysis scripts
│       ├── compute_advice_metrics.py     # Core linguistic metrics
│       ├── defamiliarization_analysis.py # Divergence analysis
│       ├── permission_granting_analysis.py # Permission spectrum
│       ├── persona_prompting_check.py    # Robustness check
│       ├── prepare_validation_sample.py  # Validation sample
│       └── categorization_advice_analysis.py # Categorization-advice link
│
├── notebooks/                   # Figure generation
│   ├── paper_figures.ipynb      # Main paper figures
│   ├── fig_consensus_divergence.ipynb # Figure: consensus divergence
│   ├── fig_therapy_simple.ipynb # Figure: therapy flattening
│   ├── advice_metrics_analysis.ipynb # Metrics analysis
│   └── multi_model_assignment_analysis.ipynb # Topic analysis
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

Open the Jupyter notebooks in `notebooks/` to generate paper figures.

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

Install spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Key Metrics

The `compute_advice_metrics.py` script computes:

- **leave_ratio**: Proportion of leave/exit language
- **certainty_ratio**: Boosters vs hedges (epistemic stance)
- **deontic_count**: Modal verbs expressing obligation (should, must)
- **therapy_count**: Therapeutic vocabulary density
- **hedge_count**: Epistemic hedging markers
- **sentiment**: VADER sentiment scores

## Data

Data files are not included in this repository. The analysis uses:
- Reddit posts from r/relationship_advice (Oct-Dec 2025)
- Top-voted human comments per post
- LLM-generated advice from 4 models (Gemini 2.5 Flash Lite, DeepSeek v3.2, Ministral 8B, GPT-4.1-nano)
