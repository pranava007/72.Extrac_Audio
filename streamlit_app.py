import os, json, pandas as pd, streamlit as st
from ReelFuntionRR import (
    download_reel, extract_audio_from_video, audio_to_tamil_text,
    get_completion_from_messages, repair_json_like, preprocess_json_to_row_dataframe,
    tamil_to_tanglish_auto, tamil_json_to_tanglish_auto
)
import requests
from dotenv import load_dotenv
load_dotenv()


def append_to_sheet(video_link: str, text: str) -> dict:
    payload = {"videoLink": video_link.strip(), "text": text}
    r = requests.post("https://script.google.com/macros/s/AKfycbw_b0vobbcvOsLNYziW0oORyVsRGX1YTnxAYIqxHucFwTZditQiANsTlxLQMexXk35A/exec", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

st.set_page_config(page_title="English", page_icon="🎥", layout="centered")
st.title("English")

# --- Inputs ---
url = st.text_input("Enter Instagram Reel URL:", placeholder="https://www.instagram.com/reel/XXXXXX/")
# with st.expander("Optional: Instagram login (reduces 403/rate-limit)"):
#     ig_user = st.text_input("IG Username", value="")
#     ig_pass = st.text_input("IG Password", value="", type="password")

col1, col2, col3, col4 = st.columns(4)
dl  = col1.button("Download Reel")
xa  = col2.button("Extract Audio")
tr  = col3.button("English-Text")
clf = col4.button("Classify + English")

# --- Step 1: Download ---
if dl:
    try:
        video = download_reel(url) #, ig_user, ig_pass)
        st.session_state["video_file"] = video
        st.success(f"Downloaded: {video}")
        st.video(video)
    except Exception as e:
        st.error(str(e))

# --- Step 2: Audio ---
if xa and "video_file" in st.session_state:
    try:
        audio = extract_audio_from_video(st.session_state["video_file"], "audio.wav")
        st.session_state["audio_file"] = audio
        st.audio(audio)
        st.success("Audio extracted.")
    except Exception as e:
        st.error(str(e))
elif xa:
    st.warning("Download a reel first.")

# # --- Step 3: Transcribe ---
# if tr and "audio_file" in st.session_state:
#     text_ta = audio_to_tamil_text(st.session_state["audio_file"])
#     st.session_state["tamil_text"] = text_ta
#     st.text_area("Tamil Transcription", text_ta, height=150)
# elif tr:
#     st.warning("Extract audio first.")

# --- Step 3: Transcribe ---
if tr and "audio_file" in st.session_state:
    text_ta = audio_to_tamil_text(st.session_state["audio_file"])
    st.session_state["tamil_text"] = text_ta
    st.text_area("English Transcription", text_ta, height=150)

    # ⬇️ Append ONLY Reel URL + extracted text to Google Sheet
    if url.strip():
        try:
            resp = append_to_sheet(url, text_ta)
            st.success(f"Saved to Google Sheet (S.No: {resp.get('serial')})")
        except Exception as e:
            st.error(f"Google Sheet append failed: {e}")
    else:
        st.warning("Enter the Reel URL to save it to Google Sheet.")
elif tr:
    st.warning("Extract audio first.")


# --- Step 4: Classify + Tanglish ---
if clf and "tamil_text" in st.session_state:
    delimiter = "####"
    system_message = (
        "You will be provided with YouTube/Instagram shorts/reels text in Tamil. "
        'Classify into "Hook","Bulid up","Body","call to action". '
        "Every classification must be present (directly or indirectly) with brief explanation. "
        "Return JSON list with only those keys."
    )
    messages = [
        {"role":"system","content":system_message},
        {"role":"user","content": f"{delimiter}{st.session_state['tamil_text']}{delimiter}"}
    ]

    with st.spinner("Calling GPT..."):
        raw = get_completion_from_messages(messages)
        st.subheader("Model output")
        st.code(raw, language="json")

        fixed = repair_json_like(raw)
        rows = preprocess_json_to_row_dataframe(fixed)
        df = pd.DataFrame(rows)
        st.subheader("Parsed classification")
        st.dataframe(df, use_container_width=True)

        with st.spinner("Tamil → Tanglish JSON..."):
            tjson = tamil_json_to_tanglish_auto(fixed)
        st.subheader("Tanglish JSON")
        st.code(tjson, language="json")

        st.download_button("Download JSON", data=fixed, file_name="classification.json", mime="application/json")
        st.download_button("Download CSV", data=df.to_csv(index=False), file_name="classification.csv", mime="text/csv")
elif clf:
    st.warning("Transcribe first.")
