import torch
import os

# Ensure the models directory exists
os.makedirs("models\sound", exist_ok=True)

# Download the Silero TTS model
torch.hub.download_url_to_file(
    "https://models.silero.ai/models/tts/ru/v4_ru.pt",
    "models/sound/silero_tts.pt"
)