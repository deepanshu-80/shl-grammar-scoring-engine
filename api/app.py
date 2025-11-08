from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile
from src.infer import predict_score

app = FastAPI()

@app.post("/score")
async def score(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    score, text = predict_score(tmp_path)
    return {"score": score, "transcript": text}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
