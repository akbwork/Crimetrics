# Crimetrics Installation & Setup Guide

## Prerequisites

- Python 3.12 or higher installed
- [Ollama](https://ollama.com/download) installed and running

## Installation

1. **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd Crimetrics
    ```

2. **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download Required Ollama Models**

    Ensure Ollama is running, then download the required models:
    ```bash
    ollama run deepseek-r1:latest
    ollama run llama3:8b
    ollama run llama3:latest
    ```

## Setup Instructions

1. **Update Absolute Paths**

    - Search the project files for any occurrence of `ananthakrishna` in absolute paths.
    - Replace `ananthakrishna` with your own username or the correct path for your system.

    For example, update:
    ```
    /Users/ananthakrishnab/Desktop/Projects/Crimmetrics_Abi/Crimetrics/...
    ```
    to match your local directory structure.

2. **Run the Application**
    ```bash
    python -m streamlit run streamlit/Home.py
    ```

    This will launch the Crimetrics web application in your browser.

## Troubleshooting

- Ensure all dependencies are installed.
- Ensure Ollama is installed, running, and the required models are downloaded.
- Double-check all file paths for correctness after replacing `ananthakrishna`.

---
For further assistance, refer to the project documentation or contact the maintainer.