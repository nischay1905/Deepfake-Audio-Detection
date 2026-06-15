import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile

# -------------------------
# Load Model
# -------------------------

model = tf.keras.models.load_model(
    "models/deepfake_audio_model_20k.keras"
)

# -------------------------
# Feature Extraction
# -------------------------

def extract_mel(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=16000
    )

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    mel_db = librosa.util.fix_length(
        mel_db,
        size=128,
        axis=1
    )

    mel_db = mel_db[:128, :128]

    mel_db = np.expand_dims(
        mel_db,
        axis=-1
    )

    mel_db = np.repeat(
        mel_db,
        3,
        axis=-1
    )

    mel_db = np.expand_dims(
        mel_db,
        axis=0
    )

    return mel_db


# -------------------------
# Streamlit UI
# -------------------------

st.title("🎙️ Deepfake Audio Detection")

st.write(
    "Upload an audio file to determine whether it is Genuine or Deepfake."
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav","mp3"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".wav"
    ) as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    features = extract_mel(temp_path)

    pred = model.predict(features)[0][0]

    confidence = max(
        pred,
        1-pred
    ) * 100

    if pred > 0.5:

        st.error(
            f"🤖 Deepfake Audio\n\nConfidence: {confidence:.2f}%"
        )

    else:

        st.success(
            f"🎤 Genuine Human Speech\n\nConfidence: {confidence:.2f}%"
        )
