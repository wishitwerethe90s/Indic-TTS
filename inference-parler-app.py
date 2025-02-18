from flask import Flask, request, send_file
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration
import torch
import soundfile as sf
from io import BytesIO

app = Flask(__name__)

# Initialize model and tokenizers once
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    try:
        # Get input data from request
        data = request.json
        description = data.get('description')
        prompt = data.get('prompt')

        if not description or not prompt:
            return {"error": "Missing 'description' or 'prompt' in request"}, 400

        # Tokenize inputs
        inputs = description_tokenizer(description, return_tensors="pt").to(device)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate audio
        generation = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask
        )

        # Create in-memory audio file
        audio_buffer = BytesIO()
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(audio_buffer, audio_arr, model.config.sampling_rate, format='WAV')
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='generated_speech.wav'
        )

    except Exception as e:
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)