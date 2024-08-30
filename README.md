Here’s a sample README file to help users set up and run the chatbot from your repository. This README assumes the code provided is stored in a GitHub repository and that users will be setting it up locally.

---

# Emergency Chatbot for Dr. Adrin

## Overview

This repository contains a Python-based chatbot designed to assist users with medical emergencies or leave a message for Dr. Adrin. The chatbot leverages Qdrant for vector-based search and GPT-Neo for generating responses when the database doesn't have a solution.

## Features

- **Emergency Handling:** Provides immediate next steps based on emergency descriptions.
- **Message Handling:** Allows users to leave messages for Dr. Adrin.
- **Local Setup:** Runs locally using Python.

## Prerequisites

- Python 3.7 or later
- `pip` for installing Python packages
- Access to a Qdrant instance (local or remote)
- Internet connection for downloading models

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/aryan0931/ Emergency-Chatbot-
cd your-repo
```

### 2. Create a Virtual Environment (Optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies

Ensure you have `pip` installed, then run:

```bash
pip install -r requirements.txt
```

### 4. Set Up Qdrant

If you don't have Qdrant installed locally, you can start a Qdrant Docker container using:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

If using a remote Qdrant instance, replace `":memory:"` in the code with your Qdrant server URL.

### 5. Configure Models

The code uses the following models:
- **Sentence Transformers:** For embedding emergency descriptions.
- **GPT-Neo:** For generating responses.

Ensure the models are available by running:

```python
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPTNeoForCausalLM

# Load models
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
gpt_neo_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
```

## Running the Chatbot

To start the chatbot, execute:

```bash
python chatbot.py
```

### Usage

1. The chatbot will prompt you to specify if you're experiencing an emergency or want to leave a message.
2. If it's an emergency, describe the issue, and the chatbot will provide immediate steps based on a Qdrant search or fallback to GPT-Neo if necessary.
3. If it's a message, provide the message, and the chatbot will confirm it has been forwarded.

## Troubleshooting

- **Model Not Found:** Ensure you have an active internet connection to download the models.
- **Qdrant Connection Issues:** Verify that your Qdrant server is running and accessible.
- **Python Errors:** Ensure all dependencies are correctly installed and you're using the recommended Python version.

## Contributing

Feel free to open issues or submit pull requests. Contributions are welcome!



