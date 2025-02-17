from diffusers import AutoPipelineForText2Image
import torch

# Load the base model
pipeline = AutoPipelineForText2Image.from_pretrained("/home/qargo/projects/ai_version_1.0.0/models/visual/paint/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')

# Load the uncensored LoRA weights
pipeline.load_lora_weights('/home/qargo/projects/ai_version_1.0.0/models/visual/flux-lora-uncensored', weight_name='lora.safetensors')

torch.backends.cudnn.benchmark = True

# Generate an image with an uncensored NSFW prompt
image = pipeline('a fox', height=64, width=64).images[0]
image.show()
