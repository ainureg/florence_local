# Базовый образ с CUDA 12.1 и cuDNN 8 (очень стабильный для torch 2.5.x)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Установка Python 3.11 + нужных утилит
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Виртуальное окружение строго на 3.11
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Обновляем pip и ставим wheel/ninja заранее (ускоряет flash-attn если придётся собирать)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

# PyTorch 2.5.1 + cu121 (точно как у тебя локально)
RUN pip install --no-cache-dir \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Основные зависимости Florence-2
# transformers 4.48+ обычно хорошо работает с Florence-2
RUN pip install --no-cache-dir \
    transformers==4.48.0 \
    accelerate==1.0.1 \
    einops \
    timm \
    opencv-python-headless \
    fastapi \
    uvicorn[standard]

# Пытаемся поставить flash-attn без сборки (самый частый рабочий способ в Docker)
# Если не найдёт wheel — pip попробует собрать (в образе есть nvcc, g++, cuda toolkit runtime)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    echo "flash-attn failed to install without build, but continuing..."

# Если выше не сработало — можно добавить явный wheel (пример для 3.11 + cu121 + torch ~2.5)
# Раскомментируй и подставь реальный URL, если нужно (смотри ниже как найти)
# RUN pip install --no-cache-dir \
#     https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/.../flash_attn-2.6.3+cu121torch2.5-cp311-cp311-linux_x86_64.whl

WORKDIR /app
COPY ./app /app

EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN pip install python-multipart

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]