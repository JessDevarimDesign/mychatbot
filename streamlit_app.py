import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

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
def load_document(file):
    from langchain.document_loaders import UnstructuredFileLoader
    loader = UnstructuredFileLoader(file)
    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


if __name__ == "__main__":

    import os

     # Show title and description.
    st.title("💬 Chatbot")
    st.write(
        "This is a simple chatbot that uses OpenAI's GPT-4o-mini model to generate responses. "
        "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
        "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
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
        uploaded_files = st.file_uploader('Upload any file format with text to analyze:', accept_multiple_files=True)

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

        if uploaded_files and 'vs' in st.session_state: #and is_api_key_valid(api_key):

            # user's question text input widget
            q = st.text_input('Ask one or more questions about the content of the uploaded data:', key='text_input')
            if q: # if the user entered a question and hit enter
                if 'vs' in st.session_state: # if vector store exists in the session state
                    vector_store = st.session_state.vs
                    answer = ask_and_get_answer(vector_store, q, k) #TODO ADD CONVERSATIONAL CONTEXT FED TO LLM

                    # text area widget for the LLM answer with flexible height
                    st.text_area('LLM Answer: ', value=answer, height=200)