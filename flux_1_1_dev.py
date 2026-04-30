import torch
from diffusers import FluxPipeline

# 加载模型并移至GPU
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")  # 将模型移至GPU以加速推理

# 启用注意力切片以减少显存占用（如果需要）
# pipe.enable_attention_slicing()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0)  # 使用CUDA生成器
).images[0]
image.save("flux-dev.png")
