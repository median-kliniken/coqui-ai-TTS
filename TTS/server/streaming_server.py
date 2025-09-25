from fastapi import FastAPI
from pydantic import BaseModel 
import uvicorn, io, time, wave, base64, torch
from fastapi import Body
from fastapi.responses import StreamingResponse
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
from pathlib import Path
original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI()

#CHUNK_MS = 200       
#SAMPLE_RATE = 22050
class StreamingInputs(BaseModel):
    #speaker_embedding: List[float]
    #gpt_cond_latent: List[List[float]]
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = '200'
this_file = Path(__file__).resolve()          
project_root = this_file.parent.parent.parent     
models_path = project_root / "tts_models" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v1.1"
models_path_str = str(models_path)
reference_voice_path = project_root/"output"/"xtts_v2.wav"  #this the target reference voice
config = XttsConfig()
config.load_json(models_path/"config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=models_path_str, use_deepspeed=False)
#model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=reference_voice_path)
t0 = time.time()
def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()
def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def predict_streaming_generator(parsed_input: StreamingInputs):
  add_wav_header = parsed_input.add_wav_header
  chunks = model.inference_stream(
        parsed_input.text,
        parsed_input.language,
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=True
    )
  for i, chunk in enumerate(chunks):
        chunk = postprocess(chunk)
        if i == 0 and add_wav_header:
            yield encode_audio_common(b"", encode_base64=False)
            yield chunk.tobytes()
        else:
            yield chunk.tobytes()
print("Inference...")
@app.post("/v1/audio/speech")
async def stream_endpoint(parsed_input: StreamingInputs = Body(...)):
    return StreamingResponse(
        predict_streaming_generator(parsed_input),
        media_type="audio/wav",
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=1234)
