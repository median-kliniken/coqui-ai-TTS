import os
import time
import torch
import torchaudio
import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import pathlib
from pathlib import Path
import torch
from importlib import reload

original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

this_file = Path(__file__).resolve()          
project_root = this_file.parent.parent.parent     
models_path = project_root / "tts_models" / "tts" / "tts_models--multilingual--multi-dataset--xtts_v1.1"

models_path_str = str(models_path)

print(models_path)      # Path object, nice representation
print(models_path_str)  # plain string
print("Loading model... from ", models_path)
config = XttsConfig()
config.load_json(models_path/"config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=models_path_str, use_deepspeed=False)
#model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["output.wav"])

print("Inference...")
t0 = time.time()
chunks = model.inference_stream(
    "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    "en",
    gpt_cond_latent,
    speaker_embedding
)

wav_chuncks = []
for i, chunk in enumerate(chunks):
    if i == 0:
        print(f"Time to first chunck: {time.time() - t0}")
    print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
    wav_chuncks.append(chunk)
wav = torch.cat(wav_chuncks, dim=0)
torchaudio.save("xtts_streaming.wav", wav.squeeze().unsqueeze(0).cpu(), 24000)