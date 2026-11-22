import numpy as np
import whisper
from funasr import AutoModel
from huggingface_hub import snapshot_download

import emotion_config
import whisper_config


emotion_config.EMOTION_MODEL_PATH.mkdir(parents=True, exist_ok=True)
whisper_config.DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Загрузка модели whisper
model_whisper = whisper.load_model(
    whisper_config.MODEL_NAME,
    device=whisper_config.DEVICE,
    download_root=whisper_config.DOWNLOAD_ROOT,
)

# Скачивание модели в заданный директорий
snapshot_download(
    repo_id=emotion_config.EMOTION_MODEL,
    local_dir=str(emotion_config.EMOTION_MODEL_PATH),
)

# Загрузка модели emotion2vec из заданного директория
model_emotion = AutoModel(
    model=str(emotion_config.EMOTION_MODEL_PATH),
    hub=emotion_config.EMOTION_HUB,
    device=emotion_config.EMOTION_DEVICE,
    ncpu=emotion_config.EMOTION_NCPU,
    disable_pbar=emotion_config.EMOTION_DISABLE_PBAR,
)


def get_transcription(audio: np.ndarray, fs: int) -> str:
    """Получает массив отсчетов и частоты дискретизации,
    возвращает извлеченный текст.
    """
    _check_audio(audio, fs)

    result = model_whisper.transcribe(
        audio,
        language=whisper_config.LANGUAGE,
        task=whisper_config.TASK,
        fp16=whisper_config.FP16,
        temperature=whisper_config.TEMPERATURE,
        condition_on_previous_text=whisper_config.CONDITION_ON_PREVIOUS_TEXT,
        initial_prompt=whisper_config.INITIAL_PROMPT,
        carry_initial_prompt=whisper_config.CARRY_INITIAL_PROMPT,
    )
    return result["text"]


def get_emotion(audio: np.ndarray, fs: int) -> dict:
    """Получает массив отсчетов и частоту дискретизации,
    возвращает словарь вида [название эмоции]:[оценка в долях от нейросети].
    """
    _check_audio(audio, fs)

    result = model_emotion.generate(
        input=audio,
        fs=fs,
        granularity=emotion_config.EMOTION_GRANULARITY,
        extract_embedding=emotion_config.EMOTION_EXTRACT_EMBEDDING,
    )[0]

    best_index = int(np.argmax(result["scores"]))
    return {
        "emotion": result["labels"][best_index].split("/")[-1],
        "score": float(result["scores"][best_index]),
    }


def _check_audio(audio: np.ndarray, fs: int) -> None:
    """Проверка аудиомассива перед формированием результата."""
    if audio.size == 0:
        raise ValueError("Array empty")

    if not isinstance(audio, np.ndarray):
        raise TypeError("Массив отсчетов должен быть объектом numpy.ndarray")

    if audio.ndim != 1:
        raise ValueError("Массив отсчетов должен быть одномерный (моно)")

    if not np.all(np.isfinite(audio)):
        raise ValueError(
            "Массив отсчетов содержит NaN или бесконечные значения"
        )

    if fs != emotion_config.EMOTION_SAMPLE_RATE:
        raise ValueError(
            f"Частота дискретизации должна быть "
            f"{emotion_config.EMOTION_SAMPLE_RATE} Гц, получено {fs} Гц"
        )