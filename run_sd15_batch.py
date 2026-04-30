from diffusers import StableDiffusionPipeline
import torch

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompts = [
    'On a fog-covered bathroom mirror, someone has clearly written the Chinese characters "深空AI实验室" with their finger. Below, in smaller text, is written the basic principle of the diffusion model: "Diffusion models are a class of generative models whose core idea is adding noise and removing noise. During training, Gaussian noise is gradually added to real images until they become pure random noise, while the reverse process is learned at each step. During generation, starting from random noise, the model gradually removes the noise under the guidance of conditional information (such as text), eventually restoring a clear image that matches the target distribution."',
    'A red apple is placed on top of a blue square box. Directly to the right side of the blue box, there is a green ceramic mug. A professional athlete is climbing the blue box.',
    'An excavated Sanxingdui bronze mask is being exhibited in front of the Pearl Shoal Waterfall in Jiuzhaigou.',
    'A highly traditional Chinese ink wash painting with blank space in the composition, depicting an ancient person meditating while wearing a modern VR headset.',
]

for i, prompt in enumerate(prompts, 1):
    print(f"[SD1.5] Generating image {i}/4...")
    image = pipe(prompt, height=640, width=480).images[0]
    output_path = f"/root/zyh/Qwen-Image/sd15_en_prompt_{i}.png"
    image.save(output_path)
    print(f"[SD1.5] Saved {output_path}")

print("[SD1.5] All done!")
