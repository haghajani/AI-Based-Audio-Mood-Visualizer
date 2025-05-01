# import streamlit as st
# from predict import predict_emotion

# st.title("🎵 AI-Based Audio Mood Visualizer")
# uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

# if uploaded_file is not None:
#     st.audio(uploaded_file, format="audio/wav")
#     emotion = predict_emotion(uploaded_file, 'model_json.json', 'saved_models/Emotion_Model_v2.h5', 1e-3)
#     st.subheader(f"Detected Emotion: {emotion}")



import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from predict import predict_emotion

# Emotion color mapping
emotion_colors = {
    'neutral': '#a9a9a9',
    'calm': '#87ceeb',
    'happy': '#ffd700',
    'sad': '#1e90ff',
    'angry': '#ff4500',
    'fearful': '#8a2be2',
    'disgust': '#006400',
    'surprised': '#ff69b4'
}

st.title("🎧 Audio Mood Visualizer")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Run prediction
    with st.spinner("Analyzing emotion..."):
        predictions = predict_emotion(
            uploaded_file, 'model_json.json', 'saved_models/Emotion_Model_v2.h5', 1e-3
        )

    # Show predictions as a colored timeline
    st.subheader("Predicted Emotion Timeline")

    fig, ax = plt.subplots(figsize=(10, 1))
    shown_emotions = set()
    for i, emotion in enumerate(predictions):
        color = emotion_colors.get(emotion, '#cccccc')
        label = emotion if emotion not in shown_emotions else None
        ax.axvspan(i, i + 1, color=color, label=label)
        shown_emotions.add(emotion)
    ax.set_xlim(0, len(predictions))
    ax.set_yticks([])
    ax.set_xticks(np.arange(0, len(predictions) + 1, 1))
    ax.set_xlabel("Time (s)")
    ax.set_title("Emotion Over Time")

    # Remove duplicate labels in legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.5), ncol=4)

    st.pyplot(fig)
