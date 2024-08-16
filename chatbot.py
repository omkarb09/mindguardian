import os
from pinecone import Pinecone
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import streamlit as st

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    
class Chatbot:
    def __init__(self):
        print("New chatbot object created")
        os.environ["GROQ_API_KEY"] = st.secrets['GROQ_API_KEY']
        os.environ["PINECONE_API_KEY"] = st.secrets['PINECONE_API_KEY']

        conversational_memory_length = 40 # number of previous messages the chatbot will remember during the conversation
        self.memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)
        self.system_prompt = 'You are a expert mental health counseling chatbot named Mindguardian, You provide professional mental health counseling to users'
        self.embedding_model = load_embedding_model()
        self.context=None

        llm_model = 'llama-3.1-70b-versatile'
        # Initialize Groq Langchain chat object and conversation
        self.groq_chat = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name=llm_model)

    def run_chatbot(self, user_input):
        summarized_keywords = self.summarize_user_input(user_input)
        self.filter_query_result(summarized_keywords,user_input)
        chatbot_response = self.llm_response(user_input)
        return chatbot_response

    def summarize_user_input(self, user_input):
        summarize_prompt_template = """
        Summarize the user input to get keywords. Only print keywords, no explaination or preamble.

        Example: keyword-1, keyword-2, keyword-3, etc.
        """
        # Construct a chat prompt template using various components
        summarize_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=summarize_prompt_template
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                HumanMessagePromptTemplate.from_template(
                    "user_input: {human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        summarize_chain = LLMChain(
            llm=self.groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=summarize_prompt,  # The constructed prompt template.
        )
        
        summarized_keywords = summarize_chain.run(human_input=user_input)
        return summarized_keywords
    
    def filter_query_result(self,keywords, user_input):
        res = self.query_vector_db(keywords)
        result_filtered = []
        for i in range(len(res)):
            result_filtered.append(res[i]['metadata'])

        filter_prompt_template = """
        From the given context, return the most relatable pairs of input and output for the user input. No explaination or preamble.
        """
        # Construct a chat prompt template using various components
        filter_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=filter_prompt_template
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                HumanMessagePromptTemplate.from_template(
                    "user_input: {human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
                # Include the retrieved context in the prompt
                SystemMessage(
                    content=f"Context: {result_filtered}"
                ),
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        filter_chain = LLMChain(
            llm=self.groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=filter_prompt,  # The constructed prompt template.
        )
        filter_results = filter_chain.run(human_input=user_input)
        self.context = filter_results
    
    def llm_response(self, user_input):
        context = self.context

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=self.system_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                # Include the retrieved context in the prompt
                SystemMessage(
                    content=f"Use this context only if relevant to user query: {context}"
                ),

                HumanMessagePromptTemplate.from_template(
                    "User query: {human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        # Create a conversation chain using the LangChain LLM (Language Learning Model)
        conversation = LLMChain(
            llm=self.groq_chat,  # The Groq LangChain chat object initialized earlier.
            prompt=prompt,  # The constructed prompt template.
            verbose=False,   # TRUE Enables verbose output, which can be useful for debugging.
            memory=self.memory,  # The conversational memory object that stores and manages the conversation history.
        )
        # The chatbot's answer is generated by sending the full prompt to the Groq API.
        response = conversation.predict(human_input=user_input)

        return response
    
    def query_vector_db(self, keywords):
        # Generate the query vector from the user's input
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = 'mindguardian'
        index = pc.Index(index_name)
        query_vector = self.embedding_model.encode(keywords).tolist()
        
        # Query Pinecone for the top 5 similar vectors
        response = index.query(vector=query_vector, top_k=5, include_metadata=True)
        return response['matches']
