from diffusers import AutoPipelineForText2Image
import torch

# Load the base model
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')

# Load the uncensored LoRA weights
pipeline.load_lora_weights('enhanceaiteam/Flux-uncensored', weight_name='lora.safetensors')

# Generate an image with an uncensored NSFW prompt
image = pipeline('a naked cute girl').images[0]
image.show()
