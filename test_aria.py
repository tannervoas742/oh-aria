from aria.utils import *

use_flash_attn = False  # Set True to enable flash attention 2
model_path = "rhymes-ai/Aria"
image_path = "cat.png"
image_paths = ["cat.png", "dog.jpg"]
video_path = "world.mp4"
num_frames = 64

processor, model = ModelLoader.load_pretrained_model(
    model_path,
    use_flash_attn=use_flash_attn,
)

image_inputs = InputHelper.image_to_inputs(model, processor, image_path)

images_inputs = InputHelper.images_to_inputs(model, processor, image_path)

video_inputs = InputHelper.video_to_inputs(model, processor, video_path, num_frames)

print(infer(image_inputs, model, processor, use_flash_attn=use_flash_attn))
print(infer(images_inputs, model, processor, use_flash_attn=use_flash_attn))
print(infer(video_inputs, model, processor, use_flash_attn=use_flash_attn))
