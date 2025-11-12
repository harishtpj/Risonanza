import streamlit as st
import matplotlib.pyplot as plt
import librosa
from   risonanza import stress_detector as sdm
from risonanza import helpers as h

EMOJI_MAP = {
    "Neutral": "😐",
    "Calm": "😌",
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😡",
    "Fear": "😨",
    "Disgust": "🤢",
    "Surprise": "😲"
}

# Page configuration
st.set_page_config(
    page_title='Stress Detector',
    page_icon='🎤',
    layout='wide'
)

@st.cache_resource
def load_model():
    model = sdm.StressDetectorModel()
    try:
        model.load()
    except:
        import setup_dataset
        dataset = h.load_dataset("ravdess_dataset")
        model.train(*h.build_features(dataset))
        model.save()
    return model

# Model Initialization
model = load_model()

def sidebar_controls():
    """Create sidebar controls"""
    st.sidebar.title("🎛️ Controls")
    st.sidebar.divider()

    st.sidebar.subheader("Model Training")
    if st.sidebar.button("🔄 Train Model", help="Train the stress detection model with synthetic data"):
        with st.spinner("Training model...", show_time=True):
            dataset = h.load_dataset("ravdess_dataset")
            t_sc = model.train(*h.build_features(dataset))
            model.save()
        st.sidebar.success(f"Model trained successfully! Accuracy: {t_sc*100}%")

    st.sidebar.divider()

    # App info
    st.sidebar.subheader("ℹ️ App Info")
    st.sidebar.info("""
    **How to use:**
    1. Train the model first
    2. Upload an audio file
    3. View stress analysis results
    """)

def analysis_page():
    """Analysis tab content"""
    st.title("🔍 Audio Analysis")
    st.write("Upload an audio file to analyze stress levels from voice patterns.")

    col1, col2 = st.columns([1, 2])

    with col1:
        # File uploader
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"],
                                      help="Supported formats: WAV, MP3")

        if audio_file is not None:
            # Display audio player
            st.audio(audio_file, format="audio/wav")
            st.success("✅ Audio uploaded successfully!")

            # Add analysis button
            if st.button("🚀 Analyze Stress", type="primary", use_container_width=True):
                with st.spinner("Predicing emotion..."):
                    audio_file.seek(0)
                    pred, stresslvl = model.predict(audio_file)

                st.subheader("📈 Results")
                st.success(f"Prediction: {EMOJI_MAP[pred]} {pred}")
                if stresslvl > 50 and pred not in ['Neutral', 'Calm', 'Happy', 'Surprise']:
                    st.warning("⚠️ High stress detected. Consider relaxation techniques.")
                else:
                    st.info("😊 Low stress level. Keep up the good mood!")

    with col2:
        st.subheader("📊 Audio Visualization")

        if audio_file is not None:
            audio_file.seek(0)
            y, sr = librosa.load(audio_file)
            fig, ax = plt.subplots()
            librosa.display.waveshow(y, sr=sr, ax=ax)
            ax.set_title("Waveform")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
        else:
            st.info("📁 Upload an audio file to see visualizations here")


def demo_page():
    """Demo tab content"""
    st.title("🎤 Live Demo")

    st.info("""
    **This section is for live testing and demonstrations.**
    You can test the model with different voice samples and see real-time results.
    """)

    audio_value = st.audio_input("Record audio")

    if audio_value:
        st.success("Recording complete!")

        if st.button("🚀 Analyze Stress"):
            with st.spinner("Predicting Stress"):
                audio_value.seek(0)
                pred, stresslvl = model.predict(audio_value)

            st.subheader("📈 Results")
            st.success(f"Prediction: {EMOJI_MAP[pred]} {pred}")
            if stresslvl > 50 and pred not in ['Neutral', 'Calm', 'Happy', 'Surprise']:
                st.warning("⚠️ High stress detected. Consider relaxation techniques.")
            else:
                st.info("😊 Low stress level. Keep up the good mood!")



def about_page():
    st.title("ℹ️ About Stress Detector")

    st.write("""
    ## 🎤 Voice Stress Detection System

    This application uses advanced audio processing and machine learning to detect 
    stress levels from voice patterns. By analyzing various acoustic features, 
    the system can identify subtle changes in voice that may indicate stress.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔧 Technical Stack")
        st.write("""
        - **Streamlit**: Web application framework
        - **Librosa**: Audio feature extraction 
        - **Scikit-learn**: Machine learning model 
        - **NumPy**: Numerical computations
        - **Pandas**: Data handling
        """)

    with col2:
        st.subheader("📊 Features Analyzed")
        st.write("""
        - MFCC Coefficients (Mel-frequency cepstral coefficients)
        - Statistical properties (Mean, Standard Deviation)
        - Pitch and frequency analysis
        - Energy distribution (RMS)
        - Spectral characteristics
        - Delta and Delta+Delta (MFCC)
        """)
    
    st.divider()
    st.subheader("Technical Features")
    st.write("""
    - **Lightweight Machine Learning**: Uses feature extraction , classical ML models instead of heavy deep learning.
    - **Low Resource Requirements**: Runs efficiently on normal hardware, making it suitable for edge devices (robots, IoT, etc.).
    - **Privacy-Friendly**: Unlike modern corporate systems, no server storage is needed — all processing is local.
    - **Open Source & Free**: No licensing fees, making it easy to integrate into research, education, or commercial prototypes.
    - **Scalable Design**: Easy to plug into robotics, security systems, or product feedback tools because of its modular architecture.""")

    st.divider()

    st.subheader("🚀 Real-world Applications")
    st.write("""
    - **Mental Health**: Early stress detection and monitoring
    - **Workplace**: Employee wellness programs
    - **Customer Service**: Quality assurance and training
    - **Security**: Voice-based analysis systems
    - **Healthcare**: Remote patient monitoring
    """)

    st.warning("""
    **⚠️ Disclaimer**: This is a demonstration project for educational purposes. 
    For actual stress diagnosis, please consult healthcare professionals.
    """)

def main():
    """Main app function"""
    # Header
    st.title('🎤 Stress Detector')
    st.caption("Voice Stress Detection using AI and Audio Analysis")


    sidebar_controls()

    # Main tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Analysis", "🎤 Demo", "ℹ️ About"])

    with tab1:
        analysis_page()

    with tab2:
        demo_page()

    with tab3:
        about_page()

# Run the main function
if __name__ == "__main__":
    main()
