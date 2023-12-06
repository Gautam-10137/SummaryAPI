from flask import Flask, request, jsonify
import requests
from transformers import pipeline
import os
app = Flask(__name__)

LANGUAGE_TOOL_API_URL = 'https://api.languagetool.org/v2/check'

@app.route('/summarize', methods=['POST'])
def summarize_and_correct():
    try:
        data = request.get_json()
        text = data['text']

        # Perform text summarization using the Bart model from transformers
        summarizer = pipeline('summarization')
        summarized_text = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)[0]['summary_text']

        # Perform grammar correction using LanguageTool API
        grammar_params = {'text': summarized_text, 'language': 'en-US'}
        grammar_response = requests.post(LANGUAGE_TOOL_API_URL, data=grammar_params)
        grammar_data = grammar_response.json()
        grammar_matches = grammar_data.get('matches', [])

        for match in reversed(grammar_matches):
            incorrect_start = match['offset']
            incorrect_end = match['offset'] + match['length']
            replacement = match['replacements'][0]['value']
            summarized_text = summarized_text[:incorrect_start] + f'**{replacement}**' + summarized_text[incorrect_end:]

        response = {
            'original_text': text,
            'summarized_text_with_correction': summarized_text
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


