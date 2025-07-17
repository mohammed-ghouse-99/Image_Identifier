from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

app = FastAPI()

# Define the model class that matches the one used during training
class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the model architecture and weights
model = FashionClassifier()
model.load_state_dict(torch.load("model/fashion_mnist_model.pt", map_location=torch.device("cpu")))
model.eval()

# List of FashionMNIST classes
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),           # Ensure single channel
    transforms.Resize((28, 28)),      # Resize to 28x28
    transforms.ToTensor(),            # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = classes[predicted.item()]

    return JSONResponse(content={"prediction": prediction})
