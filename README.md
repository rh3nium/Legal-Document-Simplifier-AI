# Legal Document Simplifier AI Tool



**A Python script that uses a T5 transformer model to simplify complex legal documents, and store the original and simplified texts in a MongoDB database. This tool is designed to make legal jargon more accessible and provide a clear, concise summary of key document details.**

---

## 🚀 Getting Started

### Prerequisites

Before running the script, ensure you have the following installed:

* **Python 3.6+**: The core language for the script.
* **MongoDB Community Server**: The script connects to a local MongoDB instance running on the default port `27017`. You must have this service running to store documents.

### Installation & Setup

1.  **Clone the Repository**: Download the project files to your local machine.

    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment**: It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

    ```bash
    # Create the virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**: Install the required Python libraries from the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Script**: Execute the script by providing the path to a legal document file as a command-line argument. The repository includes a `legal_document.txt` for testing.

    ```bash
    python document_simplifier.py legal_document.txt
    ```

---

## 📁 Project Files

### `document_simplifier.py`

This is the main Python script. It handles:
* Connecting to MongoDB.
* Reading the input document.
* Extracting key metadata (parties, dates) using regular expressions.
* Simplifying the document's content using a pre-trained `t5-base` model.
* Storing the original and simplified texts in a MongoDB collection.

```python
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pymongo import MongoClient
import datetime
import re
import sys

# Function to connect to the MongoDB database
def get_database():
    """
    Connects to a MongoDB database.
    
    Returns:
        pymongo.database.Database: The database object.
    """
    CONNECTION_STRING = "mongodb://localhost:27017/" 
    
    try:
        # Create a connection using MongoClient
        client = MongoClient(CONNECTION_STRING)
        # 'legal_docs' - database name
        return client['legal_docs']
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

# Function to read the document from a file
def read_document_from_file(file_path):
    """
    Reads a document from a text file.
    
    Args:
        file_path (str): The path to the text file.
        
    Returns:
        str: The content of the file as a single string, or None if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

# Document simplification function using pre-trained T5 model
def simplify_document(document_text, tokenizer, model):
    """
    Args:
        document_text (str): The legal text to simplify.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        model (transformers.T5ForConditionalGeneration): The T5 model.

    Return:
        str: Simplified version of the document with metadata.
    """
    if not document_text:
        return "No text to simplify."

    issuing_party = "N/A"
    receiving_party = "N/A"
    issuing_address = "N/A"
    date = "N/A"
    termination_info = "N/A"

    # Regex to find the issuing party and its address
    issuing_match = re.search(r"party of the first part, an entity with its principal place of business at (.*?), hereinafter referred to as the \"(.*?)\"", document_text, re.IGNORECASE | re.DOTALL)
    if issuing_match:
        issuing_address = issuing_match.group(1).strip()
        issuing_party = issuing_match.group(2).strip()
    
    # Regex to find the receiving party and its address
    receiving_match = re.search(r"and the party of the second part, located at (.*?), hereinafter referred to as the \"(.*?)\"", document_text, re.IGNORECASE | re.DOTALL)
    if receiving_match:
        receiving_party = receiving_match.group(2).strip()
    # Fallback for simpler cases
    if issuing_party == "N/A" and re.search(r'vendor', document_text, re.IGNORECASE):
        issuing_party = "Vendor"
    if receiving_party == "N/A" and re.search(r'client', document_text, re.IGNORECASE):
        receiving_party = "Client"

    # Regex to find the date - looks for "on this [day] day of [month], [year]"
    date_match = re.search(r"on this (\d+)(?:st|nd|rd|th)? day of ([A-Za-z]+), (\d{4})", document_text, re.IGNORECASE)
    if date_match:
        day = date_match.group(1)
        month = date_match.group(2)
        year = date_match.group(3)
        date = f"{month} {day}, {year}"
    elif re.search(r'\d{4}-\d{2}-\d{2}', document_text):
        # Fallback for YYYY-MM-DD format
        date = re.search(r'(\d{4}-\d{2}-\d{2})', document_text).group(1)
        
    # Regex to find termination date/info
    termination_match = re.search(r"(terminated by|unless terminated by|shall continue in effect until)(.*?)(?:\.|,)", document_text, re.IGNORECASE | re.DOTALL)
    if termination_match:
        termination_info = termination_match.group(0).strip()
        termination_info = termination_info.replace("shall continue in effect", "").strip()
        if termination_info.startswith("until"):
            termination_info = termination_info.replace("until", "").strip()
    
    # Generate the simplified text using T5 model
    task_prefix = "summarize: " 
    input_text = task_prefix + document_text
    
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=1024,
        truncation=True
    )
    
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=512,
        min_length=60,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    
    simplified_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add the extracted details to the simplified content
    issuing_party_display = f"{issuing_party}"
    if issuing_address != "N/A":
        issuing_party_display += f" ({issuing_address})"
    
    final_output = (
        f"• Date of issue: {date}\n"
        f"• Issuing party: {issuing_party_display}\n"
        f"• Receiving party: {receiving_party}\n"
        f"• Approximate Time of Termination: {termination_info}\n\n"
        f"Content:\n"
        f"{simplified_text}"
    )
    return final_output


# Function to store data (original & simplified documents) in MongoDB database
def store_in_mongodb(db, original_text, simplified_text):
    """
    Args:
        db (pymongo.database.Database): The database object.
        original_text (str): The original document text.
        simplified_text (str): The simplified document text.
    """
    if db is None:
        print("Cannot store document. MongoDB connection failed.")
        return
    # Getting the collection - created automatically if it doesn't exist
    collection = db['simplified_documents']
    # Document dictionary to store data
    document_data = {
        "original_text": original_text,
        "simplified_text": simplified_text,
        "date_processed": datetime.datetime.now(),
    }
    
    try:
        # Insert the document into the collection
        result = collection.insert_one(document_data)
        print(f"Document stored successfully with _id: {result.inserted_id}")
    except Exception as e:
        print(f"Error storing document in MongoDB: {e}")


##### Main execution block #####

if __name__ == "__main__":
    # Check for the required command-line argument
    if len(sys.argv) < 2:
        print("Usage: python document_simplifier.py <path_to_legal_document.txt>")
        sys.exit(1)
    
    file_path = sys.argv[1]

    # Step 1: Load the pre-trained model and tokenizer
    print("Loading pre-trained T5 model. This may take a moment...")
    # Check for GPU and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    print("Model loaded successfully.")
    
    # Step 2: Get MongoDB database connection
    db = get_database()
    if db is not None:
        print("Connected to MongoDB.")
    
    # Step 3: Read the legal document from a file
    sample_document = read_document_from_file(file_path)

    # Step 4: Simplify and store the document in database
    if sample_document:
        print("\nORIGINAL LEGAL DOCUMENT:")
        print("-" * 70)
        print(sample_document)
        
        print("\nSimplifying the document...")
        simplified_doc = simplify_document(sample_document, tokenizer, model)
        
        print("\nSIMPLIFIED DOCUMENT:")
        print("-" * 70)
        print(simplified_doc)

        # Store the data in MongoDB if the connection is successful
        store_in_mongodb(db, sample_document, simplified_doc)
    else:
        print("Could not simplify document. Please check the input file path and its content.")
