from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import sys
import os

# Load the pre-trained model and processor
processor = AutoProcessor.from_pretrained("microsoft/git-base")
model = AutoModelForCausalLM.from_pretrained("microsoft/git-base")

# Check if the image path is provided
if len(sys.argv) != 2:
    print("Usage: python caption_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Image file '{image_path}' not found.")
    sys.exit(1)

# Open the image
image = Image.open(image_path)

# Prepare the image for the model
inputs = processor(images=image, return_tensors="pt")

# Generate the caption
outputs = model.generate(**inputs)

# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)

# Print the caption
print("Generated Caption:", caption)