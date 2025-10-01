# ReelFuntionRR.py  (keep this exact filename to match imports)
import os, re, json, requests
import instaloader
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---------- OpenAI ----------
def _openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set in environment.")
    return OpenAI(api_key=key)

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    resp = _openai_client().chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return resp.choices[0].message.content

# ---------- Instagram Reel download ----------
def download_reel(reel_url: str, ig_user: str = "", ig_pass: str = "") -> str:
    if not isinstance(reel_url, str) or "instagram.com/reel/" not in reel_url:
        raise ValueError("Invalid Instagram Reel URL.")
    clean = reel_url.split("?")[0].rstrip("/")
    shortcode = clean.split("/")[-1] or clean.split("/")[-2]

    out_dir = "reels_download"
    os.makedirs(out_dir, exist_ok=True)

    L = instaloader.Instaloader()
    if ig_user and ig_pass:
        L.login(ig_user, ig_pass)   # helps avoid 403/rate-limits

    post = instaloader.Post.from_shortcode(L.context, shortcode)
    if not post.is_video:
        raise ValueError("The post is not a video.")

    video_url = post.video_url
    out_path = os.path.join(out_dir, f"{shortcode}.mp4")
    r = requests.get(video_url)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)
    return out_path

# ---------- Video → Audio ----------
def extract_audio_from_video(video_path: str, audio_path: str = "audio.wav") -> str:
    with VideoFileClip(video_path) as vid:
        vid.audio.write_audiofile(audio_path)
    return audio_path

# ---------- Speech to Tamil text (Google Web Speech) ----------
def audio_to_tamil_text(audio_file_path: str) -> str:
    r = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as src:
        audio = r.record(src)
    try:
        return r.recognize_google(audio, language="en-US") # english
    except sr.UnknownValueError:
        return "Google Speech Recognition could not understand the audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"

# (Optional) Offline Sphinx recognizer
def transcribe_audio_with_sphinx(audio_path: str) -> str:
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as src:
            r.adjust_for_ambient_noise(src)
            audio = r.record(src)
        return r.recognize_sphinx(audio)
    except sr.UnknownValueError:
        return "Sphinx could not understand audio"
    except sr.RequestError as e:
        return f"Could not process the audio using Sphinx; {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# ---------- JSON helpers ----------
def repair_json_like(text: str) -> str:
    if not text:
        return "[]"
    text = re.sub(r'^\s*```(?:json)?\s*', '', text, flags=re.I)
    text = re.sub(r'\s*```\s*$', '', text)
    text = (text.replace("“", '"').replace("”", '"')
                .replace("’", "'").replace("‘", "'"))
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        return '[{"Hook":"","Bulid up":"","Body":"' + text.replace('"','\\"') + '","call to action":""}]'
    s = text[min(starts):]
    stack, out, in_str, esc = [], [], False, False
    for ch in s:
        out.append(ch)
        if in_str:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': in_str = False
            continue
        if ch == '"': in_str = True
        elif ch in "{[": stack.append(ch)
        elif ch in "}]":
            if stack and ((stack[-1]=="{" and ch=="}") or (stack[-1]=="[" and ch=="]")):
                stack.pop()
    if in_str: out.append('"')
    closer = {"{":"}","[":"]"}
    while stack: out.append(closer[stack.pop()])
    repaired = "".join(out).strip()
    if repaired.startswith("{") and repaired.endswith("}"):
        repaired = "[" + repaired + "]"
    return repaired

def preprocess_json_to_row_dataframe(json_string: str):
    try:
        data = json.loads(json_string)
        if isinstance(data, dict):
            data = [data]
        # return raw (Streamlit will build DF)
        return data
    except json.JSONDecodeError:
        return []

# ---------- Tamil ↔ Tanglish via GPT ----------
def tamil_to_tanglish_auto(tamil_text: str) -> str:
    system = ("Convert Tamil script into Tanglish (Tamil words in English letters). "
              "Do NOT translate meaning. Preserve punctuation and spacing. "
              "If an English word is written in Tamil, output correct English spelling.")
    msgs = [{"role":"system","content":system},{"role":"user","content":tamil_text}]
    return get_completion_from_messages(msgs)

def tamil_json_to_tanglish_auto(tamil_json_text: str) -> str:
    system = ("You will be given a JSON string. Keep the JSON structure and keys exactly the same. "
              "Convert all string values from Tamil to Tanglish. Return strictly valid JSON only.")
    out = get_completion_from_messages(
        [{"role":"system","content":system},{"role":"user","content":tamil_json_text}]
    )
    try:
        json.loads(out); return out
    except Exception:
        fix_sys = "Return the same content as strictly valid JSON. No explanations or code fences."
        return get_completion_from_messages(
            [{"role":"system","content":fix_sys},{"role":"user","content":out}]
        )

__all__ = [
    "download_reel","extract_audio_from_video","audio_to_tamil_text","transcribe_audio_with_sphinx",
    "get_completion_from_messages","repair_json_like","preprocess_json_to_row_dataframe",
    "tamil_to_tanglish_auto","tamil_json_to_tanglish_auto"
]
