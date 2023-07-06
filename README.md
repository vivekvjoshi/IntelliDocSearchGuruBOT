
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

## Configuration

Before running the script, make sure to set up the necessary API keys for Pinecone and OpenAI.

### Pinecone

1. Sign up for a Pinecone account at [pinecone.io](https://www.pinecone.io).
2. Retrieve your Pinecone API key.
3. Set the `PINECONE_API_KEY` and `PINECONE_ENV` environment variables in a `.env` file:

```plaintext
PINECONE_API_KEY=<your-pinecone-api-key>
PINECONE_ENV=<your-pinecone-environment>
```

### OpenAI

1. Sign up for an OpenAI account at [openai.com](https://www.openai.com).
2. Retrieve your OpenAI API key.
3. Set the `OPENAI_API_KEY` environment variable in the `.env
