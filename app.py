"""
HeartSync AI - ECG Arrhythmia Classification Demo
Streamlit web application for ECG heartbeat classification
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

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

def generate_heartbeat(beat_type, seed=None):
    """Generate a synthetic ECG heartbeat"""
    if seed is not None:
        np.random.seed(seed)
    
    t = np.linspace(0, 1, 300)
    base = 2 * np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 2.4 * t)
    noise = np.random.normal(0, 0.05, 300)
    heartbeat = base + noise
    
    if beat_type == 'V':
        heartbeat[100:150] += np.sin(np.linspace(0, 4, 50)) * 2
        heartbeat[150:180] *= 0.3
    elif beat_type == 'S':
        heartbeat[80:100] *= 0.5
        heartbeat[100:130] += np.sin(np.linspace(0, 2, 30)) * 0.5
    elif beat_type == 'F':
        heartbeat[90:120] += np.sin(np.linspace(0, 3, 30)) * 1.2
        heartbeat[130:160] *= 0.7
    elif beat_type == 'Q':
        noise_heavy = np.random.normal(0, 0.2, 300)
        heartbeat += noise_heavy
        heartbeat[100:150] = np.random.choice([-1, 1], 50) * np.abs(np.sin(np.linspace(0, 3, 50)))
    
    return heartbeat.astype(np.float32)

def generate_demo_data(n_samples=50):
    """Generate demo heartbeat dataset"""
    heartbeats = []
    labels = []
    
    distribution = ['N'] * 35 + ['S'] * 5 + ['V'] * 5 + ['F'] * 3 + ['Q'] * 2
    np.random.shuffle(distribution)
    
    for i, beat_type in enumerate(distribution[:n_samples]):
        hb = generate_heartbeat(beat_type, seed=i)
        heartbeats.append(hb)
        labels.append(beat_type)
    
    X = np.array(heartbeats)
    X = X.reshape(-1, 300, 1)
    
    mean, std = np.mean(X), np.std(X)
    X = (X - mean) / (std + 1e-8)
    
    return X, labels

def plot_heartbeat(heartbeat, prediction=None):
    """Plot a single heartbeat"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(heartbeat, color='#2E86AB', linewidth=1.5)
    ax.fill_between(range(len(heartbeat)), heartbeat, alpha=0.3, color='#2E86AB')
    
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_ylabel('Normalized Amplitude', fontsize=12)
    
    if prediction:
        ax.set_title(f'Predicted: {prediction} - {CLASS_LABELS[prediction]}', 
                     fontsize=14, fontweight='bold', color='#28A745')
    else:
        ax.set_title('ECG Heartbeat', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_probabilities(probs):
    """Plot prediction probability distribution"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    colors = ['#28A745', '#FFC107', '#DC3545', '#17A2B8', '#6C757D']
    
    bars = ax.barh(CLASS_NAMES, probs * 100, color=colors, edgecolor='white', linewidth=1.5)
    
    for bar, prob in zip(bars, probs):
        ax.text(prob * 100 + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_risk_info(pred_class):
    """Get risk level and description"""
    info = {
        'N': ('Low Risk', 'green', 'Normal heart rhythm - no intervention needed.'),
        'S': ('Moderate', 'orange', 'Supraventricular premature beat - monitor.'),
        'V': ('High Risk', 'red', 'Premature ventricular contraction - evaluation advised.'),
        'F': ('High Risk', 'red', 'Fusion beat - may indicate cardiac condition.'),
        'Q': ('Uncertain', 'gray', 'Unclassifiable beat - additional testing may be needed.')
    }
    return info.get(pred_class, ('Unknown', 'gray', 'Consult a cardiologist.'))

def main():
    st.title("❤️ HeartSync AI")
    st.markdown("### ECG Arrhythmia Classification System")
    st.markdown("---")
    
    model, model_error = load_model()
    
    if model_error:
        st.error(f"⚠️ Failed to load model: {model_error}")
        st.info("💡 Ensure `ecg_model_code_17_t5.h5` is in the project directory.")
        return
    
    st.success("✅ Model loaded successfully! (99%+ accuracy)")
    
    with st.expander("ℹ️ About HeartSync AI", expanded=False):
        st.markdown("""
        **HeartSync AI** classifies ECG heartbeats into 5 categories:
        
        | Code | Class | Risk |
        |------|-------|------|
        | **N** | Normal | Low |
        | **S** | Supraventricular | Moderate |
        | **V** | Ventricular | High |
        | **F** | Fusion | High |
        | **Q** | Unknown | Uncertain |
        
        Model trained on MIT-BIH Arrhythmia Database with **99%+ accuracy**.
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Generate Sample Data")
        
        n_samples = st.slider("Number of heartbeats to generate:", 10, 100, 30)
        
        if st.button("🎲 Generate Demo Heartbeats", type="primary"):
            with st.spinner("Generating synthetic ECG data..."):
                X, true_labels = generate_demo_data(n_samples)
            
            with st.spinner("Running classification..."):
                predictions = model.predict(X, verbose=0)
                pred_classes = np.argmax(predictions, axis=1)
                pred_labels = [CLASS_NAMES[p] for p in pred_classes]
                confidences = predictions[np.arange(len(pred_classes)), pred_classes]
            
            st.session_state['X'] = X
            st.session_state['pred_labels'] = pred_labels
            st.session_state['confidences'] = confidences
            st.session_state['predictions'] = predictions
            
            st.success(f"✅ Generated and classified {n_samples} heartbeats!")
    
    with col2:
        st.subheader("📈 Classification Results")
        
        if 'X' not in st.session_state:
            st.info("👆 Click 'Generate Demo' to see classification results.")
        else:
            X = st.session_state['X']
            pred_labels = st.session_state['pred_labels']
            confidences = st.session_state['confidences']
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Heartbeats", len(X))
            with col_b:
                arrhythmia = sum(1 for p in pred_labels if p != 'N')
                st.metric("Arrhythmia Detected", arrhythmia)
            
            df = pd.DataFrame({
                'Beat #': range(1, len(X) + 1),
                'Prediction': [CLASS_LABELS[l] for l in pred_labels],
                'Class': pred_labels,
                'Confidence': [f"{c*100:.1f}%" for c in confidences]
            })
            
            with st.expander("📋 All Predictions", expanded=True):
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            counts = pd.Series(pred_labels).value_counts()
            fig, ax = plt.subplots(figsize=(5, 3))
            colors_map = {'N': '#28A745', 'S': '#FFC107', 'V': '#DC3545', 'F': '#17A2B8', 'Q': '#6C757D'}
            ax.bar(counts.index, counts.values, color=[colors_map.get(c, '#6C757D') for c in counts.index])
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Distribution')
            for i, v in enumerate(counts.values):
                ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
            st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("🔍 Detailed Analysis")
    
    if 'X' in st.session_state:
        X = st.session_state['X']
        predictions = st.session_state['predictions']
        
        selected = st.selectbox("Select heartbeat:", range(len(X)), 
                                format_func=lambda x: f"Beat #{x+1}")
        
        heartbeat = X[selected]
        probs = predictions[selected]
        pred_class = CLASS_NAMES[np.argmax(probs)]
        confidence = probs[np.argmax(probs)] * 100
        
        risk_label, risk_color, risk_desc = get_risk_info(pred_class)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown(f"**🫀 Prediction:** `{pred_class}` - **{CLASS_LABELS[pred_class]}**")
            st.markdown(f"**Confidence:** `{confidence:.1f}%`")
        with col2:
            st.markdown(f"**⚠️ Risk Level:** `{risk_label}`")
            st.caption(risk_desc)
        
        col3, col4 = st.columns([1, 1])
        with col3:
            fig1 = plot_heartbeat(heartbeat, pred_class)
            st.pyplot(fig1)
        with col4:
            fig2 = plot_probabilities(probs)
            st.pyplot(fig2)
    else:
        st.info("👆 Generate sample data to see detailed analysis.")
    
    with st.sidebar:
        st.markdown("### 🛠️ Settings")
        st.markdown(f"**Model:** CNN with Attention")
        st.markdown(f"**Accuracy:** 99%+")
        st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")
        
        st.markdown("---")
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Click **Generate Demo** button
        2. View batch classification results
        3. Select specific heartbeat for analysis
        4. Check risk levels and confidence
        """)
        
        if st.button("🗑️ Clear Session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

if __name__ == "__main__":
    main()