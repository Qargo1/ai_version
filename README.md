# Big thanks to:
www.youtube.com/@Ai_Austin - for memory setup

=======
## Switching to Linux WSL2 - instructions
locate your project's dirrectory
# run 'code .'

# apt-get install git

## https://www.mindspore.cn/install/en

## conda usage can create errors for llama-cpp-python
## consider installing everything via .venv

# Install python and .venv
sudo apt update
sudo apt install python3.12 python3.12-venv
python3.12 -m venv llama-env
source llama-env/bin/activate

# some standart updates
sudo apt update
sudo apt install build-essential libopenblas-dev libomp-dev
sudo apt upgrade

# pip install -U langchain-community

# installing Qdrant from official site
https://github.com/qdrant/qdrant/releases

# prepare embeddings for long memory
pip install -U sentence-transformers
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

'''
# if you prefer conda... Install Miniconda:
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash

# Create a virtual environment, taking Python 3.12 as an example:
conda create --name .conda python=3.12
conda activate .conda
'''

# Run the following command to check the Python version.
python --version

# install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo apt-get update
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
+
mb drivers?

sudo apt-get install -y nvidia-open
or
sudo apt-get install -y cuda-drivers

# for CPU only:
pip3 install torch torchvision torchaudio

# for GPU:
CUDA 12.6
+
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
or
pip install torch if not conda?

# for plain transformers (best option for me)
pip install -U bitsandbytes

# install library for GGUF
for conda
conda install -c conda-forge libgomp

for venv just plain:
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
or if mistakes were found and neutrolized
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir --force-reinstall

# Loading a GPTQ quantized model requires: only if a going to use this type
pip install -v gptqmodel --no-build-isolation

# for git-lfs - download large files from git-hub (model)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

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
