import gradio as gr
import matplotlib.pyplot as plt
from pathlib import Path
from radar import plot_radar_chart
from qa import get_q_a_private_gpt
import numpy as np
import librosa
from stt import stt , init_model
from scoring import get_score
init_model()
path = Path(__file__).parent.absolute()


def process_answer(audio,answer_ref,code: str=""):
    print(code)
    sr, y = audio
    print(len(y))
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    
    y_resampled = librosa.resample(y,orig_sr= sr,target_sr= 16000)
    print("Length of resampled audio:", len(y_resampled))
    text = stt(y_resampled)  # Transcribe audio to text using stt function
    print(text)
    # # using chatgpt process answer
    if code != "":
        text = text + ". This is my code: " + code
    result = get_score(answer_ref,text)
    print(result)
    # # using chatgpt process answer
    return plot_radar_chart(result["Proficiency"],result["Clarity"],result["Completeness"],result["Accuracy"])



with gr.Blocks() as app:
    answer_final = ""
    with gr.Row():
        with gr.Column():
            programming_exp = gr.Dropdown(["Golang", "Python", "Rust"], label="Experiments", multiselect=True)
            programming_language = gr.Dropdown(["Golang", "Python", "Rust", "System design"], label="Interview Fields")
            programming_level =  gr.Radio(["Novice","Beginner","Intermediate","Advanced","Expert"], label="Programming Level")
            with gr.Row():
                start = gr.Button("Let's go!")
                finished = gr.Button("Result!",visible=False)
        with gr.Column():
            question = gr.Textbox(label="Question")
            answer_ref = gr.Textbox(label="Answer",visible=False)
            audio = gr.Audio(sources=["microphone"])
            code = gr.Code(language="python")
            # gr.Video()
            gallery = gr.Gallery(
        label="Result", show_label=False, elem_id="gallery"
    , columns=[1], rows=[1], object_fit="contain", height="auto")

    def start_onclick(programming_language: str,programming_level: str,programming_exp):
        print(programming_language,programming_level,programming_exp)
        result = get_q_a_private_gpt(programming_language,programming_level,programming_exp)
        print("Question:", result["q"])
        print("Answer:", result["a"])
        return result["q"], result["a"]


    start.click(fn=start_onclick,inputs=[programming_language,programming_level,programming_exp],outputs=[question,answer_ref])
    audio.stop_recording(fn=process_answer,inputs=[audio,answer_ref,code],outputs=gallery)
    

if __name__ == "__main__":
    app.launch()
