import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile

MODEL_PATH = "models/deepfake_audio_model_20k.keras"
IMG_SIZE = (128, 128)
SAMPLE_RATE = 16000

st.set_page_config(page_title="Deepfake Audio Detection", layout="centered")


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_model(MODEL_PATH)


model = get_model()


def audio_to_melspectrogram(file_path, sr=SAMPLE_RATE, img_size=IMG_SIZE):
    y, sr = librosa.load(file_path, sr=sr)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=img_size[0])
    mel_db = librosa.power_to_db(mel)

    mel_resized = tf.image.resize(mel_db[..., np.newaxis], img_size)
    mel_resized = mel_resized.numpy()

    mel_rgb = np.repeat(mel_resized, 3, axis=-1)

    return mel_rgb.astype(np.float32), mel_db


def predict(file_path):
    img, mel_db = audio_to_melspectrogram(file_path)
    img_batch = np.expand_dims(img, axis=0)
    pred = model.predict(img_batch, verbose=0)[0][0]
    label = "Deepfake (AI-Generated)" if pred >= 0.5 else "Genuine (Human Speech)"
    confidence = pred if pred >= 0.5 else 1 - pred
    return label, float(confidence), mel_db


st.title("Deepfake Audio Detection")
st.write(
    "This app classifies an audio clip as either genuine human speech or "
    "AI-generated (deepfake) audio. The model converts the audio into a "
    "Mel Spectrogram and uses a CNN (EfficientNetB0) to make the prediction."
)

if model is None:
    st.warning(
        f"Model file not found at {MODEL_PATH}. "
        "Make sure deepfake_audio_model_20k.keras is placed inside the models folder."
    )

uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac", "ogg", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Analyze", disabled=model is None):
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                label, confidence, mel_db = predict(tmp_path)

                st.write("### Result")
                st.write(f"Prediction: **{label}**")
                st.write(f"Confidence: {confidence * 100:.2f}%")

                st.write("### Mel Spectrogram")
                fig, ax = plt.subplots(figsize=(8, 4))
                img = librosa.display.specshow(mel_db, sr=SAMPLE_RATE, x_axis="time", y_axis="mel", ax=ax)
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                st.pyplot(fig)

            except Exception as e:
                st.write(f"Error processing audio: {e}")
            finally:
                os.remove(tmp_path)

st.write("---")
st.write("Model trained on the Fake-or-Real (FoR) dataset using transfer learning with EfficientNetB0.")
