from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Название модели Emotion2vec.
EMOTION_MODEL = "emotion2vec/emotion2vec_plus_base"

EMOTION_MODEL_PATH = BASE_DIR / "model" / "emotion" / "emotion2vec_plus_base"

# Откуда загружается модель:
# 'hf' — Hugging Face.
EMOTION_HUB = "hf"

# Где работает модель:
# 'cpu' — процессор;
# 'cuda' — видеокарта NVIDIA.
EMOTION_DEVICE = "cpu"

# Количество CPU-потоков для вычислений.
EMOTION_NCPU = 4

# Отключить индикатор прогресса (progress bar).
# True — не показывать индикатор;
# False — разрешить его отображение.
EMOTION_DISABLE_PBAR = True

# Уровень анализа аудио:
# 'utterance' — анализ всей записи целиком;
# 'frame' — получение эмбеддингов по временным кадрам.
EMOTION_GRANULARITY = "utterance"

# Извлекать ли эмбеддинги (числовые представления аудио).
# True — дополнительно возвращать эмбеддинги;
# False — не возвращать их.
# Для обычного определения эмоции достаточно False.
EMOTION_EXTRACT_EMBEDDING = False

# дефолтная
EMOTION_SAMPLE_RATE = 16000
