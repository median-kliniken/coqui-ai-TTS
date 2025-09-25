
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
OUTPUT_DIR="${PROJECT_ROOT}/output"
export TTS_HOME="${PROJECT_ROOT}/tts_models"
export XDG_DATA_HOME="${PROJECT_ROOT}/tts_models"
echo "🔧 project root: ${PROJECT_ROOT}"
if [[ -d "${VENV_DIR}" ]]; then
    echo "✅ virtual evironment already exits ${VENV_DIR}"
else
    echo "🔧 buid up venv ..."
    python3 -m venv "${VENV_DIR}"
    echo "✅ virtual evironment build up at ${VENV_DIR}"
fi

if [ ! -d "$TTS_HOME" ]; then
    mkdir -p $TTS_HOME
    echo "🔧 make models folder ${TTS_HOME}"
else
    echo "✅ models folder already exits ${TTS_HOME}"
fi

if [ ! -d "${OUTPUT_DIR}" ]; then
    mkdir -p ${OUTPUT_DIR}
    echo "🔧 make models folder ${OUTPUT_DIR}"
else
    echo "✅ models folder already exits ${OUTPUT_DIR}"
fi
source ${VENV_DIR}/bin/activate
pip install --upgrade pip
pip install -r "${PROJECT_ROOT}/requirements.txt"


tts --model_name "tts_models/multilingual/multi-dataset/xtts_v2" \
    --text "Hello world" \
    --out_path "${PROJECT_ROOT}/output/xtts_v2.wav" \
    --speaker_idx "Viktor Eka" \
    --language_idx "en" 

#tts --model_name "tts_models/multilingual/multi-dataset/xtts_v1.1" --list_speaker_idxs
#    --text "Hello world" \
#    --speaker_wav "$path/to/speaker/wav" \
#    --language_idx "en" 
