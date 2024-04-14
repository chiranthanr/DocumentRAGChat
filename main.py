import os
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import chainlit as cl
from langchain.chains import RetrievalQA

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, 'db')


#Setup the Retrieval QA model
rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")


def load_model():
    llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    )
    return llm


def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_mistral},
        return_source_documents=True
    )
    return qa_chain


def qa_bot():
    llm = load_model()
    DB_PATH = DB_DIR
    vectorstore = Chroma(
        persist_directory=DB_PATH,embedding_function=OllamaEmbeddings(model="mistral")
    )

    qa = retrieval_qa_chain(llm, vectorstore)
    return qa


@cl.on_chat_start
async def start():
    """
    To initialize the bot when a new chat starts
    sends a welcome message and stores the bot instance in the user session
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.cocontent = (
        "Hi welcome to the document chat application"
    )
    await welcome_message.update()
    cl.user_session.set("chain",chain)


@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.

    This asynchronous function retrieves the QA bot instance from the users session, setup a callback handler for the bot's response,
    and executes the bot's call method with the given message and callback. The bots answer and the source
    documents are extracted from the response.

    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]

    text_elements = []

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            #Create the text element mentioned in the source
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)

            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources:{', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    await cl.Message(content=answer, elements=text_elements).send()

