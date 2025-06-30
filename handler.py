import torch
from torchvision import transforms
from PIL import Image
import requests
import io

# Charger le modèle
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
model.eval()

# Préprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def handler(event):
    try:
        image_url = event["input"]["image_url"]
        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)

        vector = output.squeeze().tolist()

        return {"vector": vector}
    except Exception as e:
        return {"error": str(e)}
