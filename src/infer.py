import os
import joblib
from faster_whisper import WhisperModel
import language_tool_python
import textstat

tool = language_tool_python.LanguageTool('en-US')


whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

model = joblib.load("models/grammar_model.pkl")

def transcribe(audio_path):
    segments, info = whisper.transcribe(audio_path, beam_size=1)
    return " ".join([seg.text.strip() for seg in segments])

def extract_features(text):
    matches = tool.check(text)
    num_errors = len(matches)
    words = max(len(text.split()), 1)

    return [
        (num_errors / words) * 100,
        textstat.flesch_kincaid_grade(text),
        textstat.gunning_fog(text),
        words
    ]

def predict_score(audio_path):
    text = transcribe(audio_path)
    feats = extract_features(text)
    score = model.predict([feats])[0]
    return round(score, 3), text

if __name__ == "__main__":
    path = input("Enter path to audio .wav: ")
    score, text = predict_score(path)
    print("Predicted Score:", score)
    print("Transcript:", text)
