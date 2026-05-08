"""
HeartSync AI - ECG Arrhythmia Classification Demo
Streamlit web application for real-time ECG heartbeat classification
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import wfdb
import pywt
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile

st.set_page_config(
    page_title="HeartSync AI - ECG Arrhythmia Classification",
    page_icon="❤️",
    layout="wide"
)

CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
CLASS_LABELS = {
    'N': 'Normal Beat',
    'S': 'Supraventricular Premature Beat',
    'V': 'Premature Ventricular Contraction',
    'F': 'Fusion Beat',
    'Q': 'Unclassifiable Beat'
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "ecg_model_code_17_t5.h5")

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, None
    except Exception as e:
        return None, str(e)

def denoise_signal(data):
    """Denoise ECG signal using Discrete Wavelet Transform"""
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    threshold = (np.median(np.abs(coeffs[-1])) / 0.6745) * (np.sqrt(2 * np.log(len(coeffs[-1]))))
    for i in range(len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    return pywt.waverec(coeffs=coeffs, wavelet='db5')

def segment_heartbeat(signal, r_peak_idx, window_size=300, left_samples=99, right_samples=201):
    """Segment a single heartbeat from the ECG signal"""
    start = r_peak_idx - left_samples
    end = r_peak_idx + right_samples
    
    if start < 0 or end >= len(signal):
        return None
    
    heartbeat = signal[start:end]
    
    if len(heartbeat) == window_size:
        return heartbeat
    return None

def preprocess_ecg(record_path, channel='MLII'):
    """Load and preprocess ECG record to extract heartbeats"""
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        if channel not in record.sig_name:
            channel = record.sig_name[0]
        
        channel_idx = record.sig_name.index(channel)
        signal = record.p_signal[:, channel_idx]
        
        denoised = denoise_signal(signal)
        
        heartbeats = []
        r_peaks = []
        
        for r_peak in annotation.sample:
            heartbeat = segment_heartbeat(denoised, r_peak)
            if heartbeat is not None:
                heartbeats.append(heartbeat)
                r_peaks.append(r_peak)
        
        if len(heartbeats) == 0:
            return None, None, "No valid heartbeats found in the record."
        
        X = np.array(heartbeats)
        X = X.reshape(-1, 300, 1).astype(np.float32)
        
        mean = np.mean(X)
        std = np.std(X)
        X = (X - mean) / (std + 1e-8)
        
        return X, r_peaks, None
        
    except Exception as e:
        return None, None, f"Error processing ECG: {str(e)}"

def plot_heartbeat(heartbeat, prediction=None, probabilities=None):
    """Plot a single heartbeat with prediction info"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(heartbeat, color='#2E86AB', linewidth=1.5)
    ax.fill_between(range(len(heartbeat)), heartbeat, alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    ax.set_title('ECG Heartbeat', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if prediction is not None:
        ax.set_title(f'Predicted Class: {CLASS_LABELS[prediction]}', 
                     fontsize=14, fontweight='bold', color='#28A745')
    
    plt.tight_layout()
    return fig

def plot_prediction_distribution(probs):
    """Plot prediction probability distribution"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#28A745', '#FFC107', '#DC3545', '#17A2B8', '#6C757D']
    
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, prob in zip(bars, probs):
        ax.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_ylabel('Arrhythmia Class', fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_multiple_heartbeats(heartbeats, predictions, max_show=10):
    """Plot multiple heartbeats with their predictions"""
    n = min(len(heartbeats), max_show)
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        row = i // cols
        col = i % cols
        ax = axes[row, col]
        
        ax.plot(heartbeats[i], color='#2E86AB', linewidth=1)
        ax.set_title(f'Beat {i+1}: {predictions[i]}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
    
    for i in range(n, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Sample Heartbeats with Predictions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

def get_arrhythmia_severity(pred_class):
    """Return severity level based on prediction"""
    severity = {
        'N': ('Low Risk', 'green', 'Normal heart rhythm - no intervention needed.'),
        'S': ('Moderate', 'orange', 'Supraventricular premature beat - monitor and follow-up recommended.'),
        'V': ('High Risk', 'red', 'Premature ventricular contraction - clinical evaluation advised.'),
        'F': ('High Risk', 'red', 'Fusion beat - may indicate underlying cardiac condition.'),
        'Q': ('Uncertain', 'gray', 'Unclassifiable beat - additional testing may be required.')
    }
    return severity.get(pred_class, ('Unknown', 'gray', 'Consult a cardiologist.'))

def main():
    st.title("❤️ HeartSync AI")
    st.markdown("### ECG Arrhythmia Classification System")
    st.markdown("---")
    
    model, model_error = load_model()
    
    if model_error:
        st.error(f"⚠️ Failed to load model: {model_error}")
        st.info("💡 Please ensure the trained model file (`ecg_model_code 17_t5.h5`) is in the project directory.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    with st.expander("ℹ️ About HeartSync AI", expanded=False):
        st.markdown("""
        **HeartSync AI** is a deep learning system for automated detection and classification of cardiac arrhythmias.
        
        **Model Performance:**
        - Overall Accuracy: **99%+**
        - Trained on MIT-BIH Arrhythmia Database
        
        **Classification Categories:**
        | Code | Class | Description |
        |------|-------|-------------|
        | N | Normal | Normal heart rhythm |
        | S | Supraventricular | Early beat from upper chambers |
        | V | Ventricular | Early beat from lower chambers |
        | F | Fusion | Mixed normal and abnormal beat |
        | Q | Unknown | Unclassifiable beat |
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Dataset Information")
        info_option = st.radio(
            "Choose input method:",
            ["📁 Load .dat files (WFDB format)", "📝 Demo with sample data"],
            horizontal=True
        )
        
        if info_option == "📁 Load .dat files (WFDB format)":
            st.markdown("""
            **To use your own ECG data:**
            1. Download the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
            2. Place `.dat` files in a folder
            3. Use the file path below
            """)
            
            project_path = st.text_input(
                "Dataset folder path:",
                value="mit-bih-arrhythmia-database-1.0.0",
                help="Path to folder containing .dat files"
            )
            
            record_name = st.text_input(
                "Record name (e.g., 100, 101, etc.):",
                value="100"
            )
            
            if st.button("🔍 Load Record", type="primary"):
                if record_name:
                    record_path = os.path.join(project_path, record_name)
                    
                    with st.spinner("Processing ECG data..."):
                        X, r_peaks, error = preprocess_ecg(record_path, channel='MLII')
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        st.session_state['X'] = X
                        st.session_state['r_peaks'] = r_peaks
                        st.session_state['record_name'] = record_name
                        st.success(f"✅ Loaded {len(X)} heartbeats from record {record_name}")
        else:
            st.info("🎲 Using pre-generated demo heartbeats for demonstration.")
            
            np.random.seed(42)
            n_demo = 20
            demo_heartbeats = []
            demo_labels = []
            
            for i in range(n_demo):
                t = np.linspace(0, 1, 300)
                base = 2 * np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
                noise = np.random.normal(0, 0.05, 300)
                heartbeat = base + noise
                
                if i % 5 == 4:
                    heartbeat[100:130] += np.sin(np.linspace(0, 3, 30)) * 1.5
                    demo_labels.append('V')
                elif i % 5 == 2:
                    heartbeat[80:100] *= 0.6
                    demo_labels.append('S')
                else:
                    demo_labels.append('N')
                
                demo_heartbeats.append(heartbeat)
            
            X = np.array(demo_heartbeats, dtype=np.float32)
            X = X.reshape(-1, 300, 1)
            mean, std = np.mean(X), np.std(X)
            X = (X - mean) / (std + 1e-8)
            
            st.session_state['X'] = X
            st.session_state['r_peaks'] = list(range(n_demo))
            st.session_state['record_name'] = "Demo"
            st.success(f"✅ Generated {n_demo} sample heartbeats for demonstration")
    
    with col2:
        st.subheader("🎯 Classification Results")
        
        if 'X' not in st.session_state:
            st.info("👆 Load or generate ECG data to see classification results.")
        else:
            X = st.session_state['X']
            r_peaks = st.session_state['r_peaks']
            record_name = st.session_state.get('record_name', 'Unknown')
            
            with st.spinner("Running classification..."):
                predictions = model.predict(X, verbose=0)
                pred_classes = np.argmax(predictions, axis=1)
                pred_labels = [CLASS_NAMES[p] for p in pred_classes]
                confidences = [predictions[i, pred_classes[i]] for i in range(len(pred_classes))]
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Total Heartbeats", len(X))
            
            with col_b:
                arrhythmia_count = sum(1 for p in pred_labels if p != 'N')
                st.metric("Arrhythmia Detected", arrhythmia_count)
            
            results_df = pd.DataFrame({
                'Beat #': range(1, len(X) + 1),
                'Prediction': [CLASS_LABELS[l] for l in pred_labels],
                'Class': pred_labels,
                'Confidence': [f"{c*100:.1f}%" for c in confidences]
            })
            
            with st.expander("📋 View All Predictions", expanded=True):
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            pred_counts = pd.Series(pred_labels).value_counts()
            
            fig_count, ax_count = plt.subplots(figsize=(6, 4))
            colors = {'N': '#28A745', 'S': '#FFC107', 'V': '#DC3545', 'F': '#17A2B8', 'Q': '#6C757D'}
            bar_colors = [colors.get(c, '#6C757D') for c in pred_counts.index]
            ax_count.bar(pred_counts.index, pred_counts.values, color=bar_colors, edgecolor='white')
            ax_count.set_xlabel('Class')
            ax_count.set_ylabel('Count')
            ax_count.set_title('Prediction Distribution')
            for i, v in enumerate(pred_counts.values):
                ax_count.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
            st.pyplot(fig_count)
    
    st.markdown("---")
    
    st.subheader("🔍 Detailed Heartbeat Analysis")
    
    if 'X' in st.session_state:
        X = st.session_state['X']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_beat = st.selectbox(
                "Select heartbeat to analyze:",
                options=list(range(len(X))),
                format_func=lambda x: f"Beat #{x+1}"
            )
        
        heartbeat = X[selected_beat]
        pred_probs = model.predict(heartbeat.reshape(1, 300, 1), verbose=0)[0]
        pred_class = CLASS_NAMES[np.argmax(pred_probs)]
        confidence = np.max(pred_probs) * 100
        
        severity_label, severity_color, severity_desc = get_arrhythmia_severity(pred_class)
        
        with col2:
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                st.markdown(f"**🫀 Prediction:**")
                st.markdown(f"### {pred_class} - {CLASS_LABELS[pred_class]}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
            
            with col_c2:
                st.markdown(f"**⚠️ Risk Assessment:**")
                st.markdown(f"### {severity_label}")
                st.caption(severity_desc)
        
        fig1 = plot_heartbeat(heartbeat, pred_class, pred_probs)
        st.pyplot(fig1)
        
        fig2 = plot_prediction_distribution(pred_probs)
        st.pyplot(fig2)
        
        st.markdown("---")
        
        st.subheader("📊 Batch Analysis Visualization")
        
        if len(X) > 1:
            fig_multi = plot_multiple_heartbeats(X, pred_labels, max_show=15)
            st.pyplot(fig_multi)
    else:
        st.info("👆 Load ECG data to see detailed analysis.")
    
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        
        st.markdown("**Model Information:**")
        st.text(f"Classes: {', '.join(CLASS_NAMES)}")
        st.text(f"Input shape: (300, 1)")
        st.text(f"Accuracy: 99%+")
        
        st.markdown("---")
        
        st.markdown("### 📖 Help")
        st.markdown("""
        1. **Load Data:** Select ECG record or use demo
        2. **View Results:** Check predictions and risk levels
        3. **Analyze:** Select specific heartbeat for details
        4. **Export:** Download results as CSV
        """)
        
        if st.button("🗑️ Clear Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown("*HeartSync AI Demo*")

if __name__ == "__main__":
    main()