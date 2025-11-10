import omnilingual_asr
import gradio as gr
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import torch
import torchaudio
import numpy as np
from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs

# Initialize the ASR pipeline
pipeline = ASRInferencePipeline(
    model_card="omnilingual-asr/omnilingual-asr-small",
    device="mps" if torch.backends.mps.is_available() else "cpu",
)

def chunk_audio(audio_path, chunk_length_s=30, overlap_s=2):
    waveform, sample_rate = torchaudio.load(audio_path)
    chunk_length = chunk_length_s * sample_rate
    overlap_length = overlap_s * sample_rate

    chunks = []
    start = 0
    while start < waveform.shape[1]:
        end = start + chunk_length
        chunks.append(waveform[:, start:end])
        start += chunk_length - overlap_length
    return chunks, sample_rate

def transcribe_audio(audio_file, lang):
    if audio_file is None:
        return ""

    chunks, sample_rate = chunk_audio(audio_file)

    transcriptions = []
    for chunk in chunks:
        # The pipeline expects a file path, so we save each chunk to a temporary file
        temp_file = "temp_chunk.wav"
        torchaudio.save(temp_file, chunk, sample_rate)

        # The pipeline expects a list of languages, one for each audio file
        lang_arg = [lang] if lang else None

        transcription = pipeline.transcribe([temp_file], lang=lang_arg)
        transcriptions.append(transcription[0])

    return " ".join(transcriptions)

iface = gr.Interface(
    fn=transcribe_audio,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Dropdown(supported_langs, label="Language")
    ],
    outputs="text",
    title="Omnilingual ASR",
    description="Upload an audio file and select the language to transcribe it.",
)

if __name__ == "__main__":
    iface.launch()
