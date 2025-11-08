from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)
CORS(app)

# Load model at startup
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("./final-model")
tokenizer = AutoTokenizer.from_pretrained("./final-model")
print("Model loaded!")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Predict
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=1)

    human_prob = predictions[0][0].item()
    ai_prob = predictions[0][1].item()

    return jsonify({
        'human_percentage': round(human_prob * 100, 2),
        'ai_percentage': round(ai_prob * 100, 2),
        'prediction': 'AI-GENERATED' if ai_prob > human_prob else 'HUMAN'
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000)
