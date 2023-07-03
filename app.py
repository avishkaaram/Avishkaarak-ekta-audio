import torch
import torchaudio
import gradio as gr
from transformers import pipeline
from gtts import gTTS
import tempfile
import pygame
import time

# Initialize the speech-to-text transcriber
transcriber = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english")

# Load the pre-trained question answering model
model_name = "AVISHKAARAM/avishkaarak-ekta-hindi"
qa_model = pipeline("question-answering", model=model_name)

def answer_question(context, question=None, audio=None):
    if audio is not None:
        text = transcriber(audio)
        question_text = text['text']
    else:
        question_text = question

    qa_result = qa_model(question=question_text, context=context)
    answer = qa_result["answer"]

    tts = gTTS(text=answer, lang='en')
    audio_path = tempfile.NamedTemporaryFile(suffix=".mp3").name
    tts.save(audio_path)

    return answer, audio_path

def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

# Define the Gradio interface
context_input = gr.inputs.Textbox(label="Enter Context From Which Questions Will Be Asked")
question_input = gr.inputs.Textbox(label="Ask The Question")
# audio_input = gr.inputs.Audio(label="Question Audio", type="filepath")
audio_input = gr.inputs.Audio(label="Ask The Question Through Microphone",source="microphone", type="filepath")

output_text = gr.outputs.Textbox(label="Answer")
output_audio = gr.outputs.Audio(label="Answer Audio", type="numpy")

interface = gr.Interface(
    fn=answer_question,
    inputs=[context_input, question_input, audio_input],
    outputs=[output_text, output_audio],
    title="Question Answering",
    description="Enter a context and a question to get an answer. You can also upload an audio file with the question.",
    examples=[
        ["The capital of France is Paris.", "What is the capital of France?"],
        ["OpenAI is famous for developing GPT-3.", "What is OpenAI known for?"],
    ]
)

# Launch the Gradio interface
interface.launch()