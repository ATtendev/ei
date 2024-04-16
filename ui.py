import gradio as gr
import matplotlib.pyplot as plt
from pathlib import Path
from radar import plot_radar_chart
from qa import get_q_a_private_gpt
import numpy as np
import librosa
from stt import stt, init_model
from scoring import get_score, improve_answer

if gr.NO_RELOAD:
    init_model()

path = Path(__file__).parent.absolute()

def process_answer(audio, question, answer_ref, code, language, level):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    text = stt(y_resampled)  # Transcribe audio to text using stt function
    if code != "":
        text = text + ". This is my code: " + code
    result = get_score(answer_ref, text, language, level)
    comments = improve_answer(question, text, level)
    return plot_radar_chart(result["Proficiency"], result["Clarity"], result["Completeness"], result["Accuracy"]), comments

with gr.Blocks(css="footer{display:none !important}") as app:
    answer_final = ""
    with gr.Row():
        with gr.Column(scale=1):
            programming_exp = gr.Dropdown(["Golang", "Python", "Rust"], label="Experiments", multiselect=True)
            programming_language = gr.Dropdown(["Golang", "Python", "Rust", "System design"], label="Interview Fields")
            programming_level = gr.Radio(["Novice", "Beginner", "Intermediate", "Advanced", "Expert"], label="Programming Level")
            with gr.Row():
                start = gr.Button("Let's go!")
                finished = gr.Button("Result!", visible=False)
            gallery = gr.Gallery(
                label="Result", show_label=False, elem_id="gallery", preview=True, columns=[1], rows=[1], object_fit="contain", height="auto"
            )
        with gr.Column(scale=2):
            question = gr.Textbox(label="Question", interactive=False)
            answer_ref = gr.Textbox(label="Answer", visible=False)
            audio = gr.Audio(sources=["microphone"])
            code = gr.Code(language="python")
            suggest = gr.Text(label="Make your answer better", interactive=False)
           
    def start_onclick(programming_language: str, programming_level: str, programming_exp):
        result = get_q_a_private_gpt(programming_language, programming_level, programming_exp)
        return result["q"], result["a"]

    start.click(fn=start_onclick, inputs=[programming_language, programming_level, programming_exp], outputs=[question, answer_ref])
    audio.stop_recording(fn=process_answer, inputs=[audio, question, answer_ref, code, programming_language, programming_level], outputs=[gallery, suggest])
    
if __name__ == "__main__":
    app.launch()
