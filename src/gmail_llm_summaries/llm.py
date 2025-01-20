from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_ollama.llms import OllamaLLM as Ollama
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
import logging
import time


def prepare_email_documents(emails_df):
    """Prepare emails for vectorization"""
    documents = []
    for _, email in emails_df.iterrows():
        # Combine email metadata with content
        full_content = f"""
        Subject: {email["subject"]}
        From: {email["sender"]}

        {email["body"]}
        """
        documents.append(full_content)

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    splits = text_splitter.split_text("\n\n".join(documents))
    return splits


def create_vector_store(splits):
    """Create vector store from email chunks"""
    embeddings = OllamaEmbeddings(model="mistral")
    vectorstore = Chroma.from_texts(
        texts=splits, embedding=embeddings, persist_directory="./email_vectorstore"
    )
    return vectorstore


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_vector_store_with_monitoring(texts, progress_bar=None, batch_size=10):
    """
    Create a vector store with progress monitoring
    """
    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="mistral")

    # Split texts into batches
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    all_embeddings = []

    # Process batches with progress bar
    for i, batch in enumerate(batches):
        start_time = time.time()

        # Get embeddings for batch
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

        # Log progress
        end_time = time.time()
        batch_time = end_time - start_time
        logger.info(
            f"Processed batch of {len(batch)} documents in {batch_time:.2f} seconds"
        )
        progress_bar.progress((i + 1) / len(batches))

    # Create vector store
    logger.info("Creating Chroma vector store...")
    vector_store = Chroma.from_texts(
        texts=texts, embedding=embeddings, persist_directory="./chroma_db"
    )

    return vector_store


def analyze_themes(vectorstore):
    """Extract main themes from email collection"""
    # Create theme analysis prompt
    theme_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        Based on the following email content, identify the main themes and topics discussed.
        Group related topics together and provide a brief explanation of each theme.

        Email Content:
        {context}

        Themes:
        """,
    )

    # Setup LLM chain
    llm = Ollama(model="mistral")
    theme_chain = LLMChain(llm=llm, prompt=theme_prompt)

    # Get representative samples from vector store
    samples = vectorstore.similarity_search("", k=10)
    context = "\n\n".join([doc.page_content for doc in samples])

    # Generate themes
    themes = theme_chain.run(context)
    return themes


def generate_overview(vectorstore):
    """Generate high-level summary of email collection"""
    summary_prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        Create a comprehensive overview of this email collection. Include:
        1. Main topics and their frequency
        2. Key people or organizations mentioned
        3. Any notable patterns in communication
        4. Time-sensitive or action items that appear important

        Email Content:
        {context}

        Overview:
        """,
    )

    llm = Ollama(model="mistral")
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    samples = vectorstore.similarity_search("", k=15)
    context = "\n\n".join([doc.page_content for doc in samples])

    overview = summary_chain.run(context)
    return overview
