from flask import Flask, request, jsonify, render_template, redirect
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from Classifier_pretrained import Classifier, NUM_CLASSES, CLASSES_LIST

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Classifier(NUM_CLASSES).to(device)
model.load_state_dict(torch.load('Models/best_model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    try:
        image = Image.open(file).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Error opening file: {str(e)}'}), 400

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        smax = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(smax, 1)
        predicted_label = CLASSES_LIST[predicted_class.item()]
        confidence = smax[0][predicted_class.item()].item()

    return render_template('result.html', label=predicted_label, confidence=confidence * 100)


if __name__ == '__main__':
    app.run(debug=True)
