# Streamlit 웹 앱과 필요한 모듈들을 불러옵니다.
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

# PDF 문서에서 텍스트를 추출하는 함수입니다.
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()  # 각 페이지에서 텍스트를 추출합니다.
    return text

# 텍스트를 청크로 분할하는 함수입니다.
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)  # 텍스트를 청크로 분할합니다.
    return chunks

# 벡터 저장소를 생성하는 함수입니다.
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["api_key"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # 텍스트 청크에서 벡터 저장소를 생성합니다.
    return vectorstore

# 대화 체인을 생성하는 함수입니다.
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo",openai_api_key=st.secrets["api_key"])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)  # 벡터 저장소와 메모리를 사용하여 대화 체인을 생성합니다.
    return conversation_chain

# 사용자 입력을 처리하는 함수입니다.
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})  # 대화 체인을 사용하여 질문에 응답합니다.
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  # 사용자의 메시지를 출력합니다.
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  # 챗봇의 응답을 출력합니다.

# 메인 함수입니다.
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")  # 웹 페이지의 제목과 아이콘을 설정합니다.
    st.write(css, unsafe_allow_html=True)  # CSS를 적용합니다.
    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # 대화 체인을 초기화합니다.
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None  # 대화 기록을 초기화합니다.
    st.header("Chat with multiple PDFs :books:")  # 웹 페이지의 헤더를 출력합니다.
    user_question = st.text_input("Ask a question about your documents:")  # 사용자로부터 질문을 입력받습니다.
    if user_question:
        handle_userinput(user_question)  # 사용자의 질문을 처리합니다.
    with st.sidebar:
        st.subheader("Your documents")  # 사이드바의 서브헤더를 출력합니다.
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)  # 사용자로부터 PDF 파일을 업로드받습니다.
        if st.button("Process"):  # 사용자가 'Process' 버튼을 클릭하면
            with st.spinner("Processing"):  # 처리 중임을 알리는 스피너를 출력합니다.
                raw_text = get_pdf_text(pdf_docs)  # PDF에서 텍스트를 추출합니다.
                text_chunks = get_text_chunks(raw_text)  # 텍스트를 청크로 분할합니다.
                vectorstore = get_vectorstore(text_chunks)  # 벡터 저장소를 생성합니다.
                st.session_state.conversation = get_conversation_chain(vectorstore)  # 대화 체인을 생성합니다.

# 이 스크립트를 실행하면 main 함수가 호출됩니다.
if __name__ == '__main__':
    main()
