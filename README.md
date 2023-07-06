
# Document Processing and Question Answering Script

This script provides functionality for document processing and question answering using OpenAI models and Pinecone vector indexing.

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

3. rename .env-example into .env

## Configuration

Before running the script, make sure to set up the necessary API keys for Pinecone and OpenAI.

### Pinecone

1. Sign up for a Pinecone account at [pinecone.io](https://www.pinecone.io).
2. Retrieve your Pinecone API key.
3. Set the `PINECONE_API_KEY` and `PINECONE_ENV` environment variables in a `.env` file:

```plaintext
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_ENV=<your-pinecone-environment>
PINECONE_INDEX_NAME=<your Index Name>

```

### OpenAI

1. Sign up for an OpenAI account at [openai.com](https://www.openai.com).
2. Retrieve your OpenAI API key.
3. Set the `OPENAI_API_KEY` environment variable in the `.env


## Usage

1. **Load Documents:**
    - Use the `load_document(file)` function to load a document. Supported formats are PDF and DOCX.

2. **Split Data into Smaller Chunks:**
    - Use the `get_chunk_data(data, chunk_size=256, chunk_overlap=0)` function to split the document data into smaller chunks for processing.

3. **Print Embedding Cost:**
    - Use the `print_embedding_cost(texts)` function to calculate and print the cost of embeddings based on the number of tokens.

4. **Upload Embeddings to Vector Database:**
    - Use the `insert_or_fetch_embeddings(index_name)` function to upload embeddings to a Pinecone index.

5. **Delete Pinecone Indexes:**
    - Use the `delete_pinecone_index(index_name='all')` function to delete existing Pinecone indexes.

6. **Ask Questions and Get Answers:**
    - Use the `ask_and_get_answer(vector_store, question)` function to ask a question and retrieve the answer.

7. **Ask Questions with Memory:**
    - Use the `ask_with_memory(vector_store, question, chat_history=[])` function to have a conversation with the question-answering bot and maintain a chat history.

### Run the Application
    - Execute the `Qna.py` script to run the document processing and question answering application.

Refer to the script code for more details on the function usage and program flow.

