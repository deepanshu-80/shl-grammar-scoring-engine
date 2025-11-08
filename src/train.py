import os
import pandas as pd
from faster_whisper import WhisperModel
import language_tool_python
import textstat
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

tool = language_tool_python.LanguageTool('en-US')
whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

def transcribe(audio_path):
    segments, info = whisper.transcribe(audio_path, beam_size=1)
    return " ".join([seg.text.strip() for seg in segments])

def extract_features(text):
    matches = tool.check(text)
    num_errors = len(matches)
    words = max(len(text.split()), 1)
    return {
        "errors_per_100": (num_errors / words) * 100,
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "word_count": words
    }

def main():
    df = pd.read_csv("train_metadata.csv")  # filename,label

    rows = []
    for _, row in df.iterrows():
        audio_file = os.path.join("train_audio", row["filename"] + ".wav")
        if not os.path.exists(audio_file):
            continue

        text = transcribe(audio_file)
        feats = extract_features(text)
        feats["score"] = row["label"]
        rows.append(feats)

    data = pd.DataFrame(rows)

    X = data[["errors_per_100", "fk_grade", "gunning_fog", "word_count"]]
    y = data["score"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=250, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    print("Validation MAE:", mean_absolute_error(y_val, preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/grammar_model.pkl")

    data.to_csv("final_training_features.csv", index=False)

if __name__ == "__main__":
    main()
