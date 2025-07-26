import torch
from transformers import VitsModel, AutoTokenizer
from scipy.io.wavfile import write as write_wav
import gradio as gr
import tempfile
import torchaudio.functional as F
import librosa

# Load model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

def tts(text, speed=1.0):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        waveform = model(**inputs).waveform  # shape: (1, n_samples)

    waveform_np = waveform.squeeze().numpy().astype('float32')
    
    waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=speed)

    waveform_stretched = torch.tensor(waveform_stretched).unsqueeze(0)

    # Save to temp file using scipy
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        audio_np = waveform_stretched.squeeze().cpu().numpy()
        write_wav(f.name,16000, audio_np)
        return f.name

# Gradio interface
demo = gr.Interface(
    fn=tts,
    inputs=[
        gr.Textbox(label="Nhập văn bản tiếng Việt"),
        gr.Slider(0.25, 2.0, value=1.0, step=0.25, label="Tốc độ đọc")
    ],
    outputs=gr.Audio(label="Kết quả", type="filepath", format="wav"),
    title="Vietnamese TTS with Speed Control",
    description="Chuyển văn bản tiếng Việt thành giọng nói và điều chỉnh tốc độ đọc",
)

demo.launch()
