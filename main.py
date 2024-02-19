from diffusers import StableDiffusionPipeline
import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else 'cpu')
# torch.backends.mps.is_available() # True
# device = torch.device("mps")
# x = torch.tensor([1,2,3], device=device) # This will use MPS acceleration.


pipeline = StableDiffusionPipeline.from_pretrained('simonschoe/pokeball-machine').to(device)

prompt = input('Enter the prompt for the text-to-image: \n').lower()

image = pipeline(
    prompt,
    num_inference_steps=50,
    guidance_scale=10,
    num_images_per_prompt=1
).images[0]

image

# Assuming 'image' is a PIL image or a numpy array

# Save the image to a file
image.save("generated_image.jpg")  # You can specify the desired file format (e.g., JPEG, PNG)

# Alternatively, if 'image' is a numpy array
# You can use imageio to save the image
# import imageio
# imageio.imwrite("generated_image.jpg", image)

