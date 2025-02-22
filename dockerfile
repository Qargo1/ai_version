# Базовый образ с CUDA 12.6 и Ubuntu 22.04
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

# Установка системных зависимостей в одной команде
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 \
    liblzma-dev \
    gcc \
    g++ \
    git \
    cmake \
    libpq-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    pipewire pipewire-pulse \
    pulseaudio \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание символической ссылки для libcuda.so.1
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

# Устанавливаем Python 3.11 как основной
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Установка Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Рабочая директория
WORKDIR /app

# Копируем код проекта
COPY . .

# Обновляем pip
RUN pip install --upgrade pip

# Установка PyTorch с поддержкой CUDA 12.6
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Установка остальных зависимостей
RUN pip install -r requirements.txt
RUN pip install -U sentence-transformers
RUN pip install -U openai-whisper
RUN pip install -U langchain-community
RUN pip install -U chromadb
RUN pip install -U PyAudio
RUN pip install -U pyannote.audio
RUN pip install -U TTS
RUN pip install -U psycopg2-binary

# Установка Node.js-зависимостей для визуализации
RUN npm install

# Установка llama-cpp-python с поддержкой CUDA
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_PREFIX_PATH=/usr/local/cuda" pip install llama-cpp-python

# Запуск (временно для теста)
# CMD ["bash", "-c", "npm run dev & python main.py"]
CMD ["python", "main.py"]