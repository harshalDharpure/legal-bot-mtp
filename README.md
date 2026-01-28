# Multilingual Legal: Zero-Shot Learning for POCSO Dialogues

Research project on multilingual legal NLP for POCSO (Protection of Children from Sexual Offences) Act queries, focusing on zero-shot and few-shot learning across Hindi, English, and Code-mixed text.

## Overview

This repository contains the complete research pipeline for:
- **Multilingual Legal Dialogue Understanding**: Classification of legal queries by complexity (layman, intermediate, professional)
- **Zero-Shot Transfer Learning**: Cross-lingual transfer across Hindi, English, and Code-mixed text
- **Few-Shot Learning**: Evaluation with minimal training data (5, 10, 20, 50 examples)

## Dataset

- **Total Entries**: 1,200 dialogues (400 per language)
- **Languages**: Hindi, English, Code-mixed (Hindi-English)
- **Complexity Levels**: Layman, Intermediate, Professional
- **Turn Counts**: Buckets A (2 turns), B (3 turns), C (4 turns), D (5+ turns)

## Models

### Trained Models
- **XLM-RoBERTa-Large**: 95.24% accuracy (Hindi monolingual)
- **MuRIL-Large**: 72.62% accuracy (English monolingual)

### Performance Summary
- **Supervised Baseline**: 88.49% (XLM-RoBERTa-Large)
- **Zero-Shot Transfer**: 85-94% accuracy
- **Few-Shot Learning**: 82-85% accuracy

## Project Structure

```
legal-bot/
├── experiments/                    # Experimental data splits
│   ├── exp1_supervised_baseline/   # 80/20 train/test
│   ├── exp2_monolingual_baseline/  # Per-language splits
│   ├── exp3_zeroshot_transfer/     # Zero-shot scenarios
│   └── exp4_fewshot_learning/      # Few-shot scenarios
│
├── models/                         # Model training & evaluation
│   ├── muril_large/                # MuRIL-Large model
│   ├── xlmr_large/                 # XLM-RoBERTa-Large model
│   ├── evaluate.py                 # Evaluation script
│   ├── TRAINING_RESULTS.md         # Training results
│   └── EVALUATION_RESULTS.md        # Evaluation results
│
├── hindi_posco_dataset/            # Hindi dataset (structured)
├── code_mixed_posco_dataset/        # Code-mixed dataset
└── english_posco_dataset/           # English dataset
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r models/requirements.txt
```

### 2. Train Models
```bash
cd models/muril_large
python train.py

cd ../xlmr_large
python train.py
```

### 3. Evaluate Models
```bash
cd models
python evaluate.py
```

## Results

Detailed results are available in:
- `models/TRAINING_RESULTS.md` - Training metrics and checkpoints
- `models/EVALUATION_RESULTS.md` - Comprehensive evaluation results
- `RESEARCH_COMPLETION_SUMMARY.md` - Research summary

## Key Findings

1. **XLM-RoBERTa-Large** significantly outperforms MuRIL-Large across all experiments
2. **Zero-shot transfer** works effectively (85-94% accuracy)
3. **Few-shot learning** is viable with minimal data (5-10 examples)
4. **Transfer direction** matters: English→Hindi performs better than Hindi→English

## Citation

If you use this work, please cite:
```
Multilingual Legal: Zero-Shot Learning for POCSO Dialogues
```

## License

[Add your license here]

## Contact

[Add your contact information]
