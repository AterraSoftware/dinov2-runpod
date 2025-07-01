import torch
from torchvision import transforms
from PIL import Image
import requests
import io

# Load the DINOv2 ViT-L/14 distilled model
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
model.eval()

# Define preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def handler(event):
    try:
        image_url = event["input"].get("image_url")
        if not image_url:
            return {"error": "Missing 'image_url' in input"}

        response = requests.get(image_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image: {response.status_code}"}

        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        vector = output.squeeze().tolist()

        return {"vector": vector}

    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}
