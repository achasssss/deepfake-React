import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import cv2
import face_recognition
from torch import nn
from torchvision import models
from flask import Flask, request, jsonify
import tempfile
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Model with feature visualization
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Function to convert tensor to image
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

# Function to make prediction
sm = nn.Softmax()

def predict(model, img):
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('Confidence of prediction:', logits[:, int(prediction.item())].item() * 100)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T,
                 model.linear1.weight.detach().cpu().numpy()[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (im_size, im_size))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :])
    result = heatmap * 0.5 + img * 0.8 * 255

    # Convert result to a PIL image
    result_pil = Image.fromarray(result.astype('uint8'))
    buffered = BytesIO()
    result_pil.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return [int(prediction.item()), confidence, img_str]

# Define dataset class for validation
class ValidationDataset(Dataset):
    def __init__(self, video_path, sequence_length=60, transform=None):
        self.video_path = video_path
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        video_path = self.video_path
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        vidObj = cv2.VideoCapture(video_path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                faces = face_recognition.face_locations(image)
                try:
                    top, right, bottom, left = faces[0]
                    image = image[top:bottom, left:right, :]
                except:
                    pass
                frames.append(self.transform(image))
                if len(frames) == self.count:
                    break

        if len(frames) == 0:
            raise ValueError("Failed to extract frames from the video.")

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

@app.route('/api/predict', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    video_file = request.files['video']
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        video_file.save(tmp_file.name)
        path_to_video = tmp_file.name

    # Path to the model
    path_to_model = 'D:\\DeepFake\\DeepfakeReact\\flask-server\\models\\100_detect_model.pt'

    # Define image transformations
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create dataset and load model
    video_dataset = ValidationDataset(path_to_video, sequence_length=20, transform=train_transforms)
    model = Model(2)
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()

    # Make prediction for the video
    prediction = predict(model, video_dataset[0])

    # Return the prediction
    return jsonify({'prediction': prediction[0], 'confidence': prediction[1], 'image': prediction[2]})

if __name__ == '__main__':
    app.run(debug=True)




































