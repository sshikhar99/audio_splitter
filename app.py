import os
import gradio as gr
from scipy.io.wavfile import write, read

def save_audio_locally(audio, path):
    write(path, 44100, audio)
    return path

def execute_demucs(input_audio_path,model = 'mdx_extra_q',   device = 'cpu'  ):
  if device == 'cpu':
    os.system(f"python -m demucs.separate -n {model} -d {device} {input_audio_path} -o out")
  else:
    os.system(f"python -m demucs.separate -n {model} {input_audio_path} -o out")
  if model == 'mdx_extra_q':
    vocals = 'out/mdx_extra_q/input_audio/vocals.wav'
    bass = 'out/mdx_extra_q/input_audio/bass.wav'
    drums = 'out/mdx_extra_q/input_audio/drums.wav'
    other = 'out/mdx_extra_q/input_audio/other.wav'
  elif model == 'mdx_extra':
    vocals = 'out/mdx_extra/input_audio/vocals.wav'
    bass = 'out/mdx_extra/input_audio/bass.wav'
    drums = 'out/mdx_extra/input_audio/drums.wav'
    other = 'out/mdx_extra/input_audio/other.wav'
  else:
    vocals = 'out/mdx/input_audio/vocals.wav'
    bass = 'out/mdx/input_audio/bass.wav'
    drums = 'out/mdx/input_audio/drums.wav'
    other = 'out/mdx/input_audio/other.wav'
  return vocals, bass, drums, other

def inference(input_audio, device, model):
    rate, data = input_audio
    input_audio_path = save_audio_locally(data, "input_audio.wav")
    vocals, bass, drums, other = execute_demucs(input_audio_path, model, device)
    return vocals, bass, drums, other



title = "Audio Splitter"
description = "Music Source Separation in the Waveform Domain. To use it, simply upload your audio, or click one of the examples to load them."
article = "Music source separation is the task of isolating individual instruments from a polyphonic music track. \nThis model uses Demucs, a deep learning model for music source separation. It is trained on the MUSDB18 dataset, which consists of professionally produced music tracks\nThe model is able to separate the vocals, bass, drums, and other instruments from the input audio."
examples=[['test.mp3']]

gr.Interface(
    inference,
    inputs=[
       gr.Audio(type="numpy", label="Input Audio"),
       gr.Dropdown(["cpu", "cuda"], label="Device", value="cuda"),
       gr.Dropdown(["mdx", "mdx_extra", "mdx_extra_q"], label="Model", value="mdx_extra_q")
    ],
    outputs=[
       gr.Audio(type="filepath", label="Vocals"),
       gr.Audio(type="filepath", label="Bass"),
       gr.Audio(type="filepath", label="Drums"),
       gr.Audio(type="filepath", label="Other")
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    theme="soft",
    # css="style.css",
    ).launch()

# new version for gradio 2.0

