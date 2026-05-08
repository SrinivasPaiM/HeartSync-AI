# HeartSync AI - ECG Arrhythmia Classification

A deep learning-based system for automated detection and classification of cardiac arrhythmias from single-lead ECG signals.

## Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.61% |
| Test Accuracy | 99.05% |
| Best Model Test Accuracy | 99.08% |

## Project Structure

```
HeartSync-AI/
├── Arrhythmia_Classification_Full_and_Final_Code2.ipynb  # Main notebook with complete workflow
├── app.py                                                # Streamlit demo web application
├── requirements.txt                                      # Python dependencies
├── ecg_model_code_17_t5.h5                              # Pre-trained model
└── README.md
```

## Quick Start - Streamlit Demo

### Option 1: Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Option 2: Deploy to Streamlit Cloud (Free Hosting)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "Deploy" and select your repository
4. Set:
   - **App file**: `app.py`
   - **Python version**: `3.11`
5. Click "Deploy!"

Your app will be live at `https://yourusername-heartsync-ai.streamlit.app`

## Features

- **Demo mode**: Works without any data - generates sample heartbeats
- **Real ECG analysis**: Load MIT-BIH WFDB format records
- **Risk assessment**: Classifies heartbeats as Normal/Arrhythmia with confidence
- **Visualization**: ECG waveforms, probability distributions, batch analysis

## Classification Categories

| Code | Class | Risk Level |
|------|-------|------------|
| N | Normal Beat | Low Risk |
| S | Supraventricular | Moderate |
| V | Ventricular | High Risk |
| F | Fusion Beat | High Risk |
| Q | Unknown | Uncertain |

## MIT-BIH Dataset

Download from: [PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/)

Place `.dat` files in a folder and use the path in the Streamlit app.

## Citation

```bibtex
@software{heartsync_ai,
  title = {HeartSync AI - ECG Arrhythmia Classification},
  author = {Srinivas Pai M},
  year = {2024},
  url = {https://github.com/SrinivasPaiM/HeartSync-AI}
}
```

## License

Educational/research use only. MIT-BIH database has its own usage terms.