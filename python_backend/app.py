from flask import Flask, request, jsonify
from script import simplify_document, get_database, store_in_mongodb
import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# Load the T5 model and tokenizer once when the app starts
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 't5-base'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

@app.route('/simplify', methods=['POST'])
def simplify():
    data = request.json
    document_text = data.get('document_text')

    if not document_text:
        return jsonify({'error': 'No document text provided'}), 400

    db = get_database()
    simplified_doc = simplify_document(document_text, tokenizer, model)

    # Store the data in MongoDB if the connection is successful
    if db:
        store_in_mongodb(db, document_text, simplified_doc)

    return jsonify({'simplified_document': simplified_doc})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 5000))
