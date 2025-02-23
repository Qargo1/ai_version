# Big thanks to:
www.youtube.com/@Ai_Austin - for memory setup

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
pip install peft
pip install qdrant-client
pip install -U bitsandbytes

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

# run docker for qdrant-client
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

## Installing python 3.10
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

## install cuda if needed, not needed for vllm?
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo apt-get update
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version

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

DO $$
DECLARE
    table_name text;
BEGIN
    -- Проходим по всем таблицам в схеме 'public'
    FOR table_name IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
    LOOP
        -- Удаляем каждую таблицу с каскадным удалением зависимостей
        EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', table_name);
    END LOOP;
END $$;

CREATE TABLE conversations (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    quality TEXT
                );
CREATE TABLE user_preferences (
                    id SERIAL PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL
                );
CREATE TABLE training_data (
                    id SERIAL PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    bad_response TEXT NOT NULL,
                    right_response TEXT
                );
CREATE TABLE bot_errors (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_type TEXT NOT NULL,
                    message TEXT NOT NULL
                );
                
ALTER TABLE conversations
ADD CONSTRAINT unique_prompt_response UNIQUE (prompt, response);

CREATE OR REPLACE FUNCTION limit_conversations_per_prompt()
RETURNS TRIGGER AS $$
BEGIN
    -- Удаляем старые записи, если количество строк с таким prompt превышает 5
    DELETE FROM conversations
    WHERE prompt = NEW.prompt
    AND id NOT IN (
        SELECT id
        FROM conversations
        WHERE prompt = NEW.prompt
        ORDER BY timestamp DESC
        LIMIT 5
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_limit_conversations
AFTER INSERT ON conversations
FOR EACH ROW
EXECUTE FUNCTION limit_conversations_per_prompt();

ALTER TABLE user_preferences
ADD CONSTRAINT unique_content UNIQUE (prompt);

INSERT INTO conversations (timestamp, prompt, response, quality) VALUES (CURRENT_TIMESTAMP, 'what is my name?', 'Your name is Dima. Known online as Qargo.', 'good');
INSERT INTO conversations (timestamp, prompt, response, quality) VALUES (CURRENT_TIMESTAMP, 'What is 3355 / 15?',
'223.666667', 'good');
INSERT INTO conversations (timestamp, prompt, response, quality) VALUES (CURRENT_TIMESTAMP, 'What do i like?', 'You like Anime, cats, tech and your dreams', 'good');

INSERT INTO user_preferences (prompt, response) VALUES ('What is my name', 'Your name is Dima');
INSERT INTO user_preferences (prompt, response) VALUES ('What is your name?', 'My name is Alise, i am your girfriend, how could even forget something like this???!!!!');

\q

## For audio
sudo apt update
sudo apt install pipewire pipewire-pulse

#
sudo apt update
sudo apt install pulseaudio

# open this
mkdir -p ~/.config/pulse
nano ~/.config/pulse/client.conf

# add this to file
default-server = unix:/mnt/wslg/PulseServer

# play test sound
paplay /usr/share/sounds/alsa/Front_Center.wav

# restart
systemctl --user start pipewire
systemctl --user start pipewire-pulse

pip uninstall pyaudio
sudo apt install portaudio19-dev
pip install pyaudio

export PULSE_SERVER=unix:/mnt/wslg/PulseServer
python your_script.py

#
pip install -U openai-whisper

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg
pip install setuptools-rust

# Download model for audio recognishen this preferred language DEPRICATED
https://alphacephei.com/vosk/models

## Voice Installing
pip install git+https://github.com/openai/whisper.git

# For GUI install: 
https://www.pgadmin.org/ 
or
https://www.beekeeperstudio.io/
or
Table Plus

## Some unused libraries

# pip install gradio - not sure i need it

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
