from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer
import torch

app = Flask(__name__)

# Function to load the model
def load_model(source_lang, target_lang):
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.get_json()
    
    text = data.get("text", "").strip()
    source_lang = data.get("source_lang", "en")
    target_lang = data.get("target_lang", "es")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        print(f"Translating: {text} ({source_lang} → {target_lang})")

        # Load the translation model
        model, tokenizer = load_model(source_lang, target_lang)

        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", padding=True)

        # Perform translation
        with torch.no_grad():
            translated = model.generate(**inputs)

        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        print(f"Translated: {translated_text}")

        return jsonify({"translated_text": translated_text})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
