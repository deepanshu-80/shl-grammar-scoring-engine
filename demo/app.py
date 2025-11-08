import gradio as gr
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.infer import predict_score

def demo_fn(audio):
    score, text = predict_score(audio)
    return score, text

ui = gr.Interface(
    fn=demo_fn,
    inputs=gr.Audio(
        source="upload",  
        type="filepath",
        label=" Upload Audio"
    ),
    outputs=[
        gr.Number(label="Grammar Score (0 - 5)"),
        gr.Textbox(label="Transcribed Text", lines=6)
    ],
    title="SHL Grammar Scoring Demo",
    description="Upload an audio answer and get an estimated grammar fluency score."
)


ui.launch()
