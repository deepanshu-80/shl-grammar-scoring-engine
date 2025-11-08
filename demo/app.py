import gradio as gr
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.infer import predict_score


def demo_fn(audio):
    score, text = predict_score(audio)
    return score, text

ui = gr.Interface(
    fn=demo_fn,
    inputs=gr.Audio(type="filepath"),
    outputs=["number", "text"],
    title="SHL Grammar Scoring Demo"
)

ui.launch()
