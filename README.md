# HeartSync AI - ECG Arrhythmia Classification

A deep learning-based system for automated detection and classification of cardiac arrhythmias from single-lead ECG signals.

## Overview

HeartSync AI uses a Convolutional Neural Network (CNN) with attention mechanisms to classify ECG heartbeats into 5 categories following AAMI recommendations:
- **N**: Normal beat
- **S**: Supraventricular premature beat
- **V**: Premature ventricular contraction
- **F**: Fusion of ventricular and normal beat
- **Q**: Unclassifiable / Unknown beat

## Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.61% |
| Test Accuracy | 99.05% |
| Best Model Test Accuracy | 99.08% |

### Per-Class Performance (Best Model)

| Class | Sensitivity | Specificity | F1 Score |
|-------|-------------|-------------|----------|
| N (Normal) | 99.71% | 96.80% | 99.52% |
| S (Supraventricular) | 87.14% | 99.88% | 90.84% |
| V (Ventricular) | 97.46% | 99.81% | 97.39% |
| F (Fusion) | 78.95% | 99.93% | 84.11% |
| Q (Unknown) | 99.63% | 99.99% | 99.78% |

## Project Structure

```
HeartSync-AI/
├── Arrhythmia_Classification_Full_and_Final_Code2.ipynb  # Main notebook with complete workflow
├── ecg_model_code 17_t5.h5                            # Pre-trained model
├── mit-bih-arrhythmia-database-1.0.0/                   # Dataset (48 records, 47 patients)
└── README.md
```

## Dataset

The MIT-BIH Arrhythmia Database is used for training and evaluation:

- **Source**: [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Direct Download**: [mit-bih-arrhythmia-database-1.0.0.zip](https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip)
- **48 recordings** from 47 patients (25 men, 22 women)
- **30 minutes** per recording at **360 Hz** sampling rate
- **2 leads**: Modified Limb Lead II (MLII) and Modified Lead V1/V5

## Key Features

### 1. Signal Preprocessing
- **Denoising**: Discrete Wavelet Transform (DWT) using 'db5' wavelet at level 9
- **R-Peak Detection**: Using annotated beat locations from the dataset
- **Segmentation**: 300-sample heartbeat windows (99 left, 201 right from R-peak)

### 2. CNN Model Architecture
- 1D Convolutional layers with attention mechanism
- Batch normalization and dropout for regularization
- Multi-class classification with softmax output

### 3. LIME Explainability
- Local Interpretable Model-agnostic Explanations (LIME) for model interpretability
- Visual analysis of which time steps most influence predictions
- Critical for clinical trust and validation

## Installation

```bash
# Required packages
pip install wfdb pywt tensorflow scikit-learn matplotlib seaborn pandas lime scikit-image
```

## Usage

1. **Download the Dataset**:
   - Visit [PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)
   - Download and extract to a local directory
   - Update `project_path` in the notebook to your dataset location

2. **Open the Notebook**:
   ```bash
   jupyter notebook Arrhythmia_Classification_Full_and_Final_Code2.ipynb
   ```

3. **Run the Cells** sequentially:
   - Part A: Package installation & ECG visualization
   - Part B: Denoising, R-peak detection, segmentation
   - Part C: Dataset loading & preprocessing
   - Part D: Train-test splitting & class balancing
   - Part E: Model building and training
   - Part F: Results & LIME explainability

## Notebook Sections

| Part | Description |
|------|-------------|
| **A** | Installing packages and basic ECG visualization |
| **B** | Denoising (DWT), R-peak detection, and heartbeat segmentation |
| **C** | Loading and preprocessing the full dataset |
| **D** | Train-test splitting (80-20) with class balancing (SMOTE) |
| **E** | CNN model building, training with TensorBoard & checkpoints |
| **F** | Classification results, confusion matrix, and LIME explainability |

## Hardware Requirements

- **GPU**: Recommended (T4 or better) for training
- **RAM**: 8GB+ recommended
- **Storage**: ~150MB for dataset and model

## Model Files

- `ecg_model_code 17_t5.h5` - Pre-trained model (can be loaded directly)
- `code_17_t5_weights.weights.h5` - Best model checkpoint weights

## Citation

If you use this project in your research, please cite:

```bibtex
@software{heartsync_ai,
  title = {HeartSync AI - ECG Arrhythmia Classification},
  author = {Srinivas Pai M},
  year = {2024},
  url = {https://github.com/SrinivasPaiM/HeartSync-AI}
}
```

## References

- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [AAMI Standards for Arrhythmia Classification](https://www.aami.org/)
- [LIME: Local Interpretable Model-agnostic Explanations](https://arxiv.org/abs/1602.04938)

## License

This project is for educational and research purposes. The MIT-BIH database has its own usage terms - please refer to PhysioNet guidelines.