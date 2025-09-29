# :frog: TTS Demo Server
Before you use the server, make sure you
[install](https://github.com/idiap/coqui-ai-TTS/tree/dev#install-tts) :frog: TTS
properly and install the additional dependencies with `pip install
'coqui-tts[server]'`. Then, you can follow the steps below.

**Note:** If you install :frog:TTS using ```pip```, you can also use the ```tts-server``` endpoint on the terminal instead of the `python TTS/server/server.py` arguments.

## Example commands

List officially released models:
```bash
python TTS/server/server.py --list_models  # or
tts-server --list_models
```

Run the server with the official models:
```bash
python TTS/server/server.py --model_name tts_models/en/ljspeech/tacotron2-DCA \
       --vocoder_name vocoder_models/en/ljspeech/multiband-melgan
```

Run the server with the official models on a GPU:
```bash
CUDA_VISIBLE_DEVICES="0" python TTS/server/server.py \
    --model_name tts_models/en/ljspeech/tacotron2-DCA
    --vocoder_name vocoder_models/en/ljspeech/multiband-melgan --use_cuda
```

Run the server with a custom models:
```bash
python TTS/server/server.py --tts_checkpoint /path/to/tts/model.pth \
       --tts_config /path/to/tts/config.json \
       --vocoder_checkpoint /path/to/vocoder/model.pth \
       --vocoder_config /path/to/vocoder/config.json
```

# :frog: TTS Streaming Server

## Initialization

In terminal run
```
bash set_up.sh
```
this will help download the xtts-V2 model and create virtual environment. Of course you can also do this manually.

## Start Server

In terminal run
```
python TTS/server/streaming_server.py
```

the post raw data should be JSON-form look like:

```
{       
    "text": "Er kann auch ein Roman sein, der in der deutschen Sprache geschrieben wurde, unabhängig von der Nationalität des Autors oder der Autorin. ",
    "language": "de",
    "add_wav_header": true,
    "stream_chunk_size": "20"
}
```
