# PDF Chatbot with Groq and HuggingFace Integration

This project provides a web-based chatbot interface for interacting with PDF documents using Streamlit, Groq, and HuggingFace models. Users can upload a PDF file, and the chatbot will analyze the document and respond to queries based on the content of the PDF.

## Features

- Upload and display PDF files
- Select from multiple language models (e.g., LLaMA3 8b, Mixtral 8x7b, LLaMA3 70b)
- Query the PDF content and get responses based on the document's information
- Simple and interactive chat interface

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/Harsh-1602/Chat_with_pdf.git
    cd Chat_with_pad
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables. Create a `.env` file in the root directory with the following content:

    ```
    GROQ_API_KEY=your_groq_api_key
    LLAMA_PARSE=your_llama_parse_api_key
    ```

    Replace `your_groq_api_key` and `your_llama_parse_api_key` with your actual API keys.

## Usage

1. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload a PDF file and select a language model from the sidebar.

4. Type your query into the chat input box to interact with the PDF.

## Code Overview

- **`app.py`**: Main application script that initializes Streamlit, handles file uploads, and manages the chat interface.
- **`requirements.txt`**: List of Python dependencies.
- **`.env`**: Configuration file for API keys.

## Dependencies

- `streamlit`
- `langchain`
- `llama_index`
- `dotenv`
- `pypdf2`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [Groq](https://groq.com/) for the language model API
- [HuggingFace](https://huggingface.co/) for pre-trained embeddings

## Contact

For questions or feedback, please contact (mailto:guptaharsh0216@gmail.com).
