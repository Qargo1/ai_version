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

# Installing GPTQModel (Linux only, not in Use)
https://github.com/ModelCloud/GPTQModel
# pip install vllm (Linux only, not in Use)
# pip install optimum[onnxruntime] (Not implemented)
# pip install optimum[onnxruntime-gpu] optimum[exporters] (Not implemented)

# for CPU only:
pip3 install torch torchvision torchaudio

# for GPU:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
+
CUDA 12.6

# pip install -U langchain-community

# https://www.mindspore.cn/install/en


## Switching to Linux WSL2 - instructions
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt-get install python3.11

# install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
