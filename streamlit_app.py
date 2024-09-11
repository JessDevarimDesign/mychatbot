import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from pypdf import PdfReader

    #TODO DISPLAY AND MANAGE SESSION HISTORY OF CONVERSATION
    # # Create an OpenAI client.
    # client = OpenAI(api_key=openai_api_key)

    # # Create a session state variable to store the chat messages. This ensures that the
    # # messages persist across reruns.
    # if "messages" not in st.session_state:
    #     st.session_state.messages = []

    # # Display the existing chat messages via `st.chat_message`.
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.markdown(message["content"])

    # # Create a chat input field to allow the user to enter a message. This will display
    # # automatically at the bottom of the page.
    # if prompt := st.chat_input("What is up?"):

    #     # Store and display the current prompt.
    #     st.session_state.messages.append({"role": "user", "content": prompt})
    #     with st.chat_message("user"):
    #         st.markdown(prompt)

    #     # Generate a response using the OpenAI API.
    #     stream = client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=[
    #             {"role": m["role"], "content": m["content"]}
    #             for m in st.session_state.messages
    #         ],
    #         stream=True,
    #     )

    #     # Stream the response to the chat using `st.write_stream`, then store it in 
    #     # session state.
    #     with st.chat_message("assistant"):
    #         response = st.write_stream(stream)
    #     st.session_state.messages.append({"role": "assistant", "content": response})

# loading unstructured files of all sorts as LangChain Documents
#file is string , file name
def load_document(file):
    # from langchain.document_loaders import UnstructuredFileLoader
    # loader = UnstructuredFileLoader(file)
    # data = loader.load()
    # return data
    from langchain_core.documents import Document
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text+= page.extract_text()
    return Document(page_content=text)


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents([data])
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def get_history_aware_retriever(llm, retriever):
    from langchain.chains import create_history_aware_retriever
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.prompts import ChatPromptTemplate

    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever

def get_history_aware_rag_chain(vector_store, k=3):
    # from langchain import hub
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import MessagesPlaceholder
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    # chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # answer = chain.run(q)
    history_aware_retriever = get_history_aware_retriever(llm, retriever)

# # See full prompt at https://smith.langchain.com/hub/langchain-ai/retrieval-qa-chat
#     retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever , combine_docs_chain)
    return rag_chain

def ask_and_get_answer_history_aware(q, rag_chain):
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_community.chat_message_histories import SQLChatMessageHistory

        ### Statefully manage chat history ###
    #TODO need only one store. ALready stored chat hisotry in browswer session
    store = {}

#seems like it gets wiped out every rerun and no context to llm
#need to feed this function from the main, browswer hisotry and integrate it with ChatMessageHisotry class
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        # if session_id not in store:
        #     store[session_id] = ChatMessageHistory()
        # print(store[session_id])
        # return store[session_id]
        return SQLChatMessageHistory(session_id, "sqlite:///memory.db")
    
    # def construct_langchain_message_history(session_history):
    #     langchain_message_history=[]
    #     from langchain_core.messages import HumanMessage, AIMessage
    #     for m in session_history:
    #         if m['role']=="user":
    #             langchain_message_history.append(HumanMessage(m['content']))
    #         else:
    #             langchain_message_history.append(AIMessage(m['content']))
    #     return langchain_message_history

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history, #does it get fed and also used to store messages in the same function ?
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    answer = conversational_rag_chain.invoke(
        {"input": q},
        config={
            "configurable": {"session_id": "abc1"}
        },  # constructs a key "abc1" in `store`.
    )["answer"]
    return answer
    

if __name__ == "__main__":

    import os

     # Show title and description.
    st.title("💬 Chatbot")
    st.write(
        "This is a simple chatbot that uses OpenAI's GPT-4o-mini model to generate responses. "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
        # "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
        "This code assumes you already given a VALID openapi key"
    )

    # # Ask user for their OpenAI API key via `st.text_input`.
    # # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
    # # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
    # openai_api_key = st.text_input("OpenAI API Key", type="password")
    # if not openai_api_key:
    #     st.info("Please add your OpenAI API key to continue.", icon="🗝️")

    with st.sidebar:
        # text_input for the OpenAI API key
        api_key = st.text_input('Your OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        # # check if the API key is not valid
        # if api_key and not is_api_key_valid(api_key):
        #     st.error('Invalid OpenAI API key. Please provide a valid key.')
        #     st.stop()


        # file uploader widget
        uploaded_files = st.file_uploader('Upload pdf file format with text to analyze:', accept_multiple_files=True)

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=8192, value=512)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3)

        # initialize add_data
        add_data = False

        # # add data button widget
        # if is_api_key_valid(api_key):
        add_data = st.button('Add Data', key='add_data')
        # else:
        #     st.info('No OpenAI API key. Please provide a valid key.')

        if uploaded_files and add_data: # if the user uploaded files and clicked the add data button
            # check_openai_api_key_exist()
            with st.spinner('Reading, chunking and embedding data ...'):

                # create ./docs/ folder if it doesn't exist
                if not os.path.exists('./docs/'):
                    os.mkdir('./docs/')

                # list to store all the chunks
                all_chunks = []
                
                for uploaded_file in uploaded_files:

                    # writing the file from RAM to the current directory on disk
                    bytes_data = uploaded_file.read()
                    file_name = os.path.join('./docs/', uploaded_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'File name: {os.path.basename(file_name)}, Chunk size: {chunk_size}, Chunks: {len(chunks)}')
                    all_chunks.extend(chunks)

                # tokens, embedding_cost = calculate_embedding_cost(all_chunks)
                # st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(all_chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                st.success('Uploaded, chunked and embedded successfully.')

                # deleting files from the docs folder after they have been chunked and embedded
                for file in os.listdir('./docs/'):
                    os.remove(os.path.join('./docs/', file))

                # deleting the docs folder
                os.rmdir('./docs/')

    #main window
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
    if uploaded_files and 'vs' in st.session_state: #and is_api_key_valid(api_key):

        # user's question text input widget
        q = st.text_input('Ask one or more questions about the content of the uploaded data:', key='text_input')
        if q: # if the user entered a question and hit enter
            st.session_state.messages.append({"role": "user", "content": q})
            if 'vs' in st.session_state: # if vector store exists in the session state
                vector_store = st.session_state.vs
                rag_chain = get_history_aware_rag_chain(vector_store, k) 
                response = ask_and_get_answer_history_aware(q, rag_chain)

                # text area widget for the LLM answer with flexible height
                st.text_area('LLM Answer: ', value=response, height=200)
                # st.text_area('LLM Context Used: ', value=response['context'], height=200)
                st.session_state.messages.append({"role": "assistant", "content": response})
