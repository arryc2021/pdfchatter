# 📄 PDF Q&A with Ollama and HuggingFace Embeddings

Ask questions about your PDF documents using a locally running LLM powered by [Ollama](https://ollama.com/) and semantic search with [HuggingFace Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Built with [Streamlit](https://streamlit.io/) and [LangChain](https://www.langchain.com/).

![App Screenshot]
 
 <img width="975" height="404" alt="image" src="https://github.com/user-attachments/assets/1cf569a6-b212-46aa-964f-1e693128d39c" />

<img width="975" height="435" alt="image" src="https://github.com/user-attachments/assets/0488de32-aaca-4ae8-a5b7-0e47f6725085" />

---

## 🔧 Features

- 📥 Upload any PDF document
- 🧠 Local language model using Ollama (`llama3`)
- 🔍 Semantic chunking with HuggingFace Embeddings
- 🤖 Real-time Q&A interface via Streamlit
- 🧩 Uses LangChain for document loading, splitting, vector storage, and retrieval

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/pdf-qa-ollama.git
cd pdf-qa-ollama
2. Install dependencies
We recommend using a virtual environment.

bash
Copy
Edit
pip install -r requirements.txt
Or manually install:

bash
Copy
Edit
pip install streamlit langchain chromadb sentence-transformers ollama
3. Start Ollama (if not already running)
Make sure Ollama is installed and the llama3 model is pulled:

bash
Copy
Edit
ollama run llama3
You can choose any other supported model like mistral, gemma, etc., by changing the model name in the code.

🚀 Usage
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Then, open the provided local URL in your browser (usually http://localhost:8501).

📂 How It Works
Upload a PDF file via the Streamlit UI.

The PDF is loaded and split into manageable text chunks.

Each chunk is embedded using HuggingFace's all-MiniLM-L6-v2.

Chunks are stored in a Chroma vector database.

When you ask a question:

The app retrieves relevant chunks using vector similarity.

The question and context are passed to the Ollama LLM.

The model generates a response, shown in the UI.

