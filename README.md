# Big thanks to:
www.youtube.com/@Ai_Austin - for memory setup

# Implementing memory first step
ollama pull nomic-embed-text 
install postgreSQL and add it to the path

# For tweaking postgreSQL - in terminal running commands
psql -U postgres

CREATE USER qargo WITH PASSWORD '5787' SUPERUSER;
CREATE DATABASE memory_agent;
GRANT ALL PRIVILEGES ON SCHEMA public TO qargo;
GRANT ALL PRIVILEGES ON DATABASE memory_agent TO qargo;
\c memory_agent
CREATE TABLE conversations (
id SERIAL PRIMARY KEY,
timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
prompt TEXT NOT NULL,
response TEXT NOT NULL
);
INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'what is my name?', 'Y
our name is Dima. Known online as Qargo.');
INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'What is 3355 / 15?',
'223.666667');
INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'What do i like?', 'You like Anime, cats, tech and your dreams');
SELECT * FROM conversations;

CREATE TABLE user_preferences (
    id SERIAL PRIMARY KEY,
    key TEXT NOT NULL,          -- Например: "favorite_food", "communication_style"
    value TEXT NOT NULL,        -- Например: "пицца", "формальный"
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    quality TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

# Create a local ollama model by creating Modelfile:

FROM ./zephyr-ollama
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
TEMPLATE """{% for message in messages %}{{message['role']}}: {{message['content']}}{% endfor %}"""

# Build model package
ollama create zephyr -f ./zephyr-ollama/Modelfile

# Possible requirements
transformers
peft
bitsandbytes
pynvml
schedule
PyQt5
PyQtWebEngine
speechrecognition
pyaudio
numpy
dataclasses
urllib3
resource
psutil
openai-whisper
setuptools-rust
soundfile
gptqmodel
wheel
gekko
pandas
wget
silero
scipy
torchaudio
librosa
torchvision
noisereduce
vosk
keyboard
TTS
deepspeech
coqui-stt-model-manager
python-osc
future_fstrings
ollama
chromadb
psycopg
ast

# Implementing cuda and torch by running
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124