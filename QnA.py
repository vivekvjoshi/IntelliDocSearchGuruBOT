import os
from dotenv import load_dotenv, find_dotenv
from langchain.llms import OpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI


load_dotenv(find_dotenv(), override=True)

#load documnets
def load_document(file):
    import os
    from langchain.document_loaders import PyPDFLoader

    fileName, extention = os.path.splitext(file)
    print(f' Loading the {file}')
    
    if extention.lower() == ".pdf":
        loader = PyPDFLoader(file)
    elif extention.lower() == ".docx":
        from langchain.document_loaders import Docx2txtLoader
        Docx2txtLoader.load(file)
   
   
    else:
        print("Documnet fromat not supported")
        return None
    
    data = loader.load()
    return data

# Split the data into smaller chunks to Support OpenAPI limitation

def get_chunk_data(data,chunk_size=256,chunk_overlap=0):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    textSplitter= RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = textSplitter.split_documents(data)
    return chunks


#print the cost of embeddings

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
  
#uploading embeddings into vector database

def insert_or_fetch_embeddings(index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings
    
    embeddings = OpenAIEmbeddings()
    
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=1536, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store
    


#delete Pinecone Indexes

def delete_pinecone_index(index_name='all'):
    import pinecone
    pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))
    
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')
        
        
# Creaating Question and Answer Bot

def ask_and_get_answer(vector_store, question):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.run(question)
    return answer
    
    
def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI
    
    llm = ChatOpenAI(temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))
    
    return result, chat_history
    
    
#Running the Program

data = load_document(os.environ.get("FILE"))
#print(data[1].page_content)
chunks = get_chunk_data(data=data)

#print the cost of embeddings
print_embedding_cost(chunks)

#delete old Index ( as only one free version is supported)
#delete_pinecone_index()

#create embeddings
vector_store=insert_or_fetch_embeddings(os.environ.get("PINECONE_INDEX_NAME"))

chat_history=[]

import time
i = 1
print('Write Quit or Exit to quit.')
while True:
    q = input(f'Question #{i}: ')
    i = i + 1
    if q.lower() in ['quit', 'exit']:
        print('Quitting ... bye bye!')
        time.sleep(2)
        break
    
    result,chat_history = ask_with_memory(vector_store, q,chat_history)
    print(f'\nAnswer: {result["answer"]}')
    print(f'\n {"-" * 50} \n')