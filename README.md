# Big thanks to:
www.youtube.com/@Ai_Austin - for memory setup

# Follow all the instructions for Triton:
https://github.com/woct0rdho/triton-windows

# Then this libraries:
pip install -U bitsandbytes
pip install unsloth

## For using vllm - the fastest and interesting library Switch to Linux WSL2 - instructions:
wsl --install
wsl --install -d Ubuntu
wsl --shutdown
wsl --export Ubuntu "A:\wsl-ubuntu.tar"
wsl --unregister Ubuntu
mkdir A:\wsl-ubuntu
wsl --import Ubuntu "A:\wsl-ubuntu" "A:\wsl-ubuntu.tar" --version 2
wsl --list --verbose

locate your project's dirrectory
# run 'code .'

# apt-get install git

# some standart updates
sudo apt update
sudo apt install build-essential libopenblas-dev libomp-dev
sudo apt upgrade
=======
pip install vllm

## If we are working in standart transformers (+ don't need to do anything, never show any errors)
# Install python 3.12
https://www.python.org/downloads/release/python-3129/
## Installing venv:
python -m venv .venv
.venv\Scripts\activate

pip install -U psycopg2-binary
pip install cachetools
pip install -U sentence-transformers

# pip install qdrant-client - have problems with:
unsloth 2025.2.15 requires protobuf<4.0.0, but you have protobuf 5.29.3 which is incompatible.
unsloth-zoo 2025.2.7 requires protobuf<4.0.0, but you have protobuf 5.29.3 which is incompatible.

pip install numpy
pip install accelerate
# run docker for qdrant-client(if docker is not installed, instructions are down)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
pip install -U bitsandbytes

# Install needed cuda (12.6)
https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local
# Install pytorch for cuda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install peft

pip install typing_extensions
pip3 install torch torchvision torchaudio

pip install python-dev-tools
pip install -U TTS
pip install -U PyAudio
pip install -U openai-whisper
pip3 install -U speechbrain
pip install -U langchain-community
pip install -r requirements.txt

## Docker FAQ
# Install docker
sudo apt update
sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker qargo  # Добавить себя в группу docker
exit

# Create dockerfile in your workspace
# Create .dockerignore

# create build
nvidia-docker build -t ai-bot .

# run build
docker run --rm -it ai-bot

# работа из терминала контейнера
docker run --rm -it --gpus all --device /dev/snd -v /mnt/wslg/PulseServer:/mnt/wslg/PulseServer -e PULSE_SERVER=unix:/mnt/wslg/PulseServer ai-bot bush
при запуске pip install покеты будут запускаться, но при выходе из контейнера они пропадут

# Подключение больших файлов моделей
docker run --rm -it -v /home/qargo/projects/ai_version_1.0.0/models:/app/models ai-bot


## If you need different python via linux - Installing python 3.10:
1. **Добавь репозиторий**:
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

2. **Установи Python 3.11**:
PYTHON_CONFIGURE_OPTS="--enable-framework"
sudo apt install python3.10 python3.10-dev python3.10-venv
sudo apt install python3.10-dev

3. **Создай виртуальную среду**:
python3.10 -m venv /home/qargo/projects/ai_version_1.0.0/.venv
source /home/qargo/projects/ai_version_1.0.0/.venv/bin/activate
pip install --upgrade pip

4. **Проверь**:
python --version  # Должно показать Python 3.10.x
python -c "import _lzma; print('LZMA работает!')"

# for git-lfs - download large files from git-hub (model)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# installing Qdrant from official site
https://github.com/qdrant/qdrant/releases

# prepare embeddings for long memory
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

## For voice
git clone https://github.com/coqui-ai/TTS
make system-deps  # intended to be used on Ubuntu (Debian). Let us know if you have a different OS.
make install

## Installing PostgreSQL via terminal commands
sudo apt-get update
sudo apt install postgresql-client-common
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo apt-get install libpq-dev
sudo systemctl enable postgresql

# For tweaking postgreSQL - in terminal running commands
sudo -i -u postgres
createuser --interactive --pwprompt

createdb memory_agent
psql
GRANT ALL PRIVILEGES ON DATABASE memory_agent TO qargo;
\q

sudo systemctl status postgresql

# Open PosgreSql in terminal and create new tables
psql -U qargo -d memory_agent -h localhost
instructions in postgre_helper

## For 3d visual

# Добавляем репозиторий NodeSource
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -

# Устанавливаем Node.js и npm
sudo apt-get install -y nodejs

# Проверяем версии
node -v
npm -v

# Установка зависимостей
npm install

npm run dev
