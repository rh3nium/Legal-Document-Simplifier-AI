# Legal Document Simplifier AI Tool

<img src="LegalAI (1).png">

### A Python script that uses a T5 transformer model to simplify complex legal documents, and store the original and simplified texts in a MongoDB database. This tool is designed to make legal jargon more accessible and provide a clear, concise summary of key document details. 

---

## 🚀 Instructions to Run the Script

### Prerequisites

Before running the script, ensure you have the following installed:

* **Python 3.0+**: The core language for the script.
* **MongoDB Community Server**: The script connects to a local MongoDB instance running on the default port `27017`. You must have this service running to store documents.

### Installation & Setup

1.  **Clone the Repository**: Download the project files to your local machine after navigating to your desired location.

    ```bash
    git clone https://github.com/rh3nium/Legal-Document-Simplifier-AI.git
    cd Legal-Document-Simplifier-AI
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
    python3 script.py
    ```

---
