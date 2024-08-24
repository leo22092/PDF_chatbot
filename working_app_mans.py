import streamlit as st 
from dotenv import load_dotenv
import os
HUGGINGFACEHUB_API_TOKEN="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"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN


from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings#new line

from langchain.vectorstores import FAISS
from langchain. memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import anthropic as model
# from langchain.llms import HuggingFaceTextGenInference as modell
# from langchain.llms import gpt4all as modell
# from langchain.chat_models import ChatHuggingFace
# from langchain.chat_models import ChatHuggingFace
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers
model_name = "facebook/bart-base"  # Or any compatible model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# modell = transformers.Conversation(model=model, tokenizer=tokenizer)
# modell = ChatHuggingFace(model_name="facebook/bart-base") 

from htmlTemplates import user_template,bot_template,css
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        print(pdf)
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks=text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings=HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    
    # embeddings=HuggingFaceInstructEmbeddings(model_name="awinml/instructor-xl-embeddings")

    # embeddings=OpenAIEmbeddings
    vectorstore=FAISS.from_texts(text_chunks,embedding=embeddings)
    vectorstore.save_local("./sub/")

    return vectorstore


#############
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm=HuggingFaceHub(repo_id="google/flan-t5-small",model_kwargs={"temperature":0,"max_length":512})
# chain=load_qa_chain(llm=llm,chain_type="stuff")
# chain.run(inputvectorstore,question=query)

###########3
def get_conversation_chain(vectorstor):
    # llm=model 
    memory=ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        # retriever=vectorstor.as_retriver(),
        retriever =vectorstor,
        memory=memory
    )
    print("ger")
    return conversation_chain

def handle_userinput(user_question):
    response=st.session_state.conversation({'question': user_question})
    st.write(response)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation= None


    st.header("Contin  :books:")
    user_question=st.text_input("Ask a Question about your Requirement PDF?")
    if user_question:
        handle_userinput(user_question)
    st.write(user_template.replace("{{MSG}}","Hello poker "),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello suci"),unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader(
            "Upload your PDF s here and click process ",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get the pdf text
                raw_text=get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # Get the text chunkds
                text_chunks=get_text_chunks(raw_text)
                st.write(text_chunks)
                # Create vector store with embedding
                # vectorstore=get_vectorstore(text_chunks=text_chunks)
                embeddings=HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")

                # vectorstore=FAISS.from_texts(text_chunks,embedding=embeddings)

                print("Vectorstore created")
                
                vectorstor = FAISS.load_local(folder_path="./sub/", embeddings=embeddings,allow_dangerous_deserialization=True)
                print("Vector store loaded")
            
                # create conversation chain
                retriver=vectorstor.as_retriever()
                st.session_state.conversation = get_conversation_chain(retriver)
    st.session_state.conversation
if __name__=='__main__':
    main()