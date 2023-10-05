from langchain.embeddings.openai import OpenAIEmbeddings
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import Tool
from langchain.agents import initialize_agent



OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

# find API key in console at app.pinecone.io
YOUR_API_KEY = os.environ.get('PINECONE_API_KEY')
# find ENV (cloud region) next to API key in console
YOUR_ENV = "asia-southeast1-gcp-free"

index_name = 'salesforcedocs'
pinecone.init(
    api_key=YOUR_API_KEY,
    environment=YOUR_ENV
)

text_field = "text"

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

query = "who is lebron james"

vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)

# chat completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)
# retrieval qa chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool for all questions'
        )
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

agent(query)