import pandas as pd
import os
from infer import predict_score


def main():
    df = pd.read_csv("test_metadata.csv")
    results = []

    for _, row in df.iterrows():
        fname = row["filename"]
        audio_path = os.path.join("test_audio", f"{fname}.wav")

        try:
            score, _ = predict_score(audio_path)
        except:
            score = 0.5  # fallback score

        results.append([fname, score])

    out = pd.DataFrame(results, columns=["filename", "label"])
    out.to_csv("submission.csv", index=False)
    print("submission.csv created successfully!")

if __name__ == "__main__":
    main()
