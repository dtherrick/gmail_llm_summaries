from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import logging
import time
import os
from typing import List, Optional
from tqdm import tqdm


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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )

    splits = text_splitter.split_text("\n\n".join(documents))
    return splits


def create_vector_store(
    splits: List[str], persist_directory: str = "./email_vectorstore"
) -> Chroma:
    """Create vector store from email chunks"""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Using the latest embedding model
    )
    vectorstore = Chroma.from_texts(
        texts=splits, embedding=embeddings, persist_directory=persist_directory
    )
    return vectorstore


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_vector_store_with_monitoring(
    texts: List[str],
    progress_bar: Optional[tqdm] = None,
    batch_size: int = 10,
    persist_directory: str = "./chroma_db",
) -> Chroma:
    """
    Create a vector store with progress monitoring
    """
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
            f"Processed batch {i + 1}/{len(batches)} ({len(batch)} documents) in {batch_time:.2f} seconds"
        )

        if progress_bar:
            progress_bar.progress((i + 1) / len(batches))

    # Create vector store
    logger.info("Creating Chroma vector store...")
    vector_store = Chroma.from_texts(
        texts=texts, embedding=embeddings, persist_directory=persist_directory
    )

    return vector_store


def analyze_themes(vectorstore: Chroma) -> str:
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

    # Setup LLM chain with GPT-4
    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        temperature=0.2,  # Lower temperature for more focused analysis
    )
    theme_chain = LLMChain(llm=llm, prompt=theme_prompt)

    # Get representative samples from vector store
    samples = vectorstore.similarity_search("", k=10)
    context = "\n\n".join([doc.page_content for doc in samples])

    # Generate themes
    themes = theme_chain.run(context)
    return themes


def generate_overview(vectorstore: Chroma) -> str:
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

    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.3)
    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

    samples = vectorstore.similarity_search("", k=15)
    context = "\n\n".join([doc.page_content for doc in samples])

    overview = summary_chain.run(context)
    return overview


# Example usage
if __name__ == "__main__":
    # Ensure OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Please set the OPENAI_API_KEY environment variable")

    # Your email processing code here
    pass
