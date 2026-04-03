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
│   ├── r_relationship_advice_comments_cleaned.parquet  # Reddit comments
│   ├── advice_metrics.parquet          # Computed linguistic metrics
│   ├── permission_metrics.parquet      # Permission structure analysis
│   ├── multi_model_assignments.parquet # Topic classifications
│   ├── defamiliarization/
│   │   └── post_consensus.csv          # Per-post consensus metrics
│   ├── pairwise_validation.csv         # Validation sample pairs (Coder 1)
│   ├── validation_coder2.csv           # Validation sample pairs (Coder 2)
│   ├── interrater_agreement_results.csv # Inter-rater agreement statistics
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
│       ├── calculate_interrater_agreement.py   # Inter-rater reliability
│       └── categorization_advice_analysis.py   # Categorization-advice link
│
├── notebooks/                                  # Figure generation
│   ├── fig_consensus_divergence.ipynb          # Figure: consensus vs divergence
│   └── fig_therapy_simple.ipynb                # Figure: therapy flattening
│
├── LLM_Advice___Big_Data___Society/            # Paper LaTeX source
│   ├── main.tex                                # Main document
│   ├── abstract.tex, intro.tex, etc.           # Section files
│   ├── custom.bib                              # References
│   └── figures/                                # Paper figures
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

# Calculate inter-rater agreement for validation
python scripts/analysis/calculate_interrater_agreement.py
```

### 4. Figures

Jupyter notebooks in `notebooks/` generate paper figures.

## Requirements

```
pip install -r requirements.txt
```

## Data

Data files are stored using [Git LFS](https://git-lfs.github.com/) (Large File Storage). To clone with data:

```bash
# Install Git LFS (if not already installed)
git lfs install

# Clone repository (LFS files download automatically)
git clone https://github.com/[username]/recognition-without-prescription.git
```

### Data Files

| File | Description |
|------|-------------|
| `llm_advice.parquet` | LLM-generated advice responses from 4 models |
| `stratified_sample.parquet` | Stratified sample of Reddit posts with human comments |
| `r_relationship_advice_comments_cleaned.parquet` | Cleaned Reddit comments |
| `advice_metrics.parquet` | Computed linguistic metrics (leave ratio, certainty, hedging, etc.) |
| `permission_metrics.parquet` | Permission structure analysis results |
| `multi_model_assignments.parquet` | Topic classifications from multi-model assignment |
| `defamiliarization/post_consensus.csv` | Per-post community consensus metrics |
| `pairwise_validation.csv` | Validation sample pairs (Coder 1 annotations) |
| `validation_coder2.csv` | Validation sample pairs (Coder 2 annotations) |
| `interrater_agreement_results.csv` | Inter-rater agreement statistics |
| `persona_comparison.csv` | Persona prompting robustness check results |

### Source Data

The analysis uses:
- Reddit posts from r/relationship_advice (Oct-Dec 2025)
- Top-voted human comments per post
- LLM-generated advice from 4 models (Gemini 2.5 Flash Lite, DeepSeek v3.2, Ministral 8B, GPT-4.1-nano)

## Validation

Two independent coders evaluated 50 matched human-LLM response pairs. When both coders judged that a clear difference existed, directional agreement was:
- Certainty: 96% (24/25 cases)
- Leave-orientation: 100% (20/20 cases)
- Therapeutic framing: 100% (44/44 cases)

See `scripts/analysis/calculate_interrater_agreement.py` for details.

## License

[Add license information]

## Citation

[Add citation information when published]
