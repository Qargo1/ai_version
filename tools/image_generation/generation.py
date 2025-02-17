from diffusers import AutoPipelineForText2Image
import torch

# Load the base model
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')

# Load the uncensored LoRA weights
pipeline.load_lora_weights('/home/qargo/projects/ai_version_1.0.0/models/visual/flux-lora-uncensored', weight_name='lora.safetensors')

# Generate an image with an uncensored NSFW prompt
image = pipeline('a naked cute young girl').images[0]
image.show()
