"""
SMS Spam Detector - Streamlit App
Loads the trained model, tokenizer, and config, then lets a user
type a message and see if it's predicted as spam or ham.
"""

import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load saved files ---
model = load_model("model/model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model/config.pkl", "rb") as f:
    config = pickle.load(f)

max_len = config["max_len"]

# --- App UI ---
st.title("📩 SMS Spam Detector")
st.write("Type a message below and I'll tell you if it looks like spam or not.")

text = st.text_input("Enter a message:")

if st.button("Check Message"):
    if text.strip() == "":
        st.warning("Please type a message first.")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
        prob = model.predict(padded, verbose=0)[0][0]

        if prob > 0.5:
            st.error(f"🚫 This looks like SPAM ({prob:.1%} confidence)")
        else:
            st.success(f"✅ This looks like HAM (not spam) ({(1 - prob):.1%} confidence)")
