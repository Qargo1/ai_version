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
=======
## Switching to Linux WSL2 - instructions
locate your project's dirrectory
# run 'code .'

# apt-get install git

## https://www.mindspore.cn/install/en

# Install Miniconda:
cd /tmp
curl -O https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.10.3-Linux-$(arch).sh
bash Miniconda3-py37_4.10.3-Linux-$(arch).sh -b
cd -
. ~/miniconda3/etc/profile.d/conda.sh
conda init bash

# Create a virtual environment, taking Python 3.11.11 as an example:
conda create -n mindspore_py39 python=3.11.11 -y
conda activate mindspore_py39

# Run the following command to check the Python version.
python --version

# To activate this environment, use
conda activate mindspore_py39

# Install GCC 9.
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-9 -y

# Installing MindSpore
export MS_VERSION=2.5.0
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/${MS_VERSION}/MindSpore/unified/x86_64/mindspore-${MS_VERSION/-/}-cp311-cp311-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple

## Installation Verification
# run:
python -c "import mindspore;mindspore.set_device(device_target='CPU');mindspore.run_check()"

# The outputs should be the same as:
MindSpore version: __version__
The result of multiplication calculation is correct, MindSpore has been installed on platform [CPU] successfully!

# install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2404-12-6-local_12.6.2-560.35.03-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2404-12-6-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
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

# fir git-lfs - download large files from git-hub (model)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# pip install -U langchain-community

# pip install gradio - not sure i need it

## Some unused libraries

# Installing GPTQModel (Linux only, not in Use) - потребуют переписывания части кода и специфического окружения.
https://github.com/ModelCloud/GPTQModel
pip install -v gptqmodel --no-build-isolation

# pip: compile and install
# You can install optional modules like autoround, ipex, vllm, sglang, bitblas, and ipex.
# Example: pip install -v --no-build-isolation .[vllm,sglang,bitblas,ipex,auto_round]
pip install -v . --no-build-isolation

# pip install vllm (Linux only, not in Use) - должен ускорить работу, потребуют переписывания части кода и специфического окружения.
# pip install optimum[onnxruntime] (Not implemented) - должен быть совместим с моим кодом
# pip install optimum[onnxruntime-gpu] optimum[exporters] (Not implemented)
