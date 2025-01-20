import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict
from auth import get_gmail_service
from get_starred_emails import get_starred_emails
from llm import (
    prepare_email_documents,
    create_vector_store,
    analyze_themes,
    generate_overview,
    create_vector_store_with_monitoring,
)


def extract_email_details(service, message) -> Dict:
    """Extract relevant details from email message"""
    msg = (
        service.users()
        .messages()
        .get(userId="me", id=message["id"], format="full")
        .execute()
    )
    headers = msg["payload"]["headers"]

    # Extract headers
    subject = next(
        (h["value"] for h in headers if h["name"].lower() == "subject"), "No Subject"
    )
    sender = next(
        (h["value"] for h in headers if h["name"].lower() == "from"), "Unknown"
    )
    try:
        date_str = next(
            (h["value"] for h in headers if h["name"].lower() == "date"), ""
        )
        date = pd.to_datetime(date_str).strftime("%Y-%m-%d") if date_str else "Unknown"
    except Exception as e:
        date = "Unknown"

    # Extract body (simplified version - you might want to handle different payload types)
    body = ""
    if "parts" in msg["payload"]:
        for part in msg["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                body = part["body"].get("data", "")
    else:
        body = msg["payload"]["body"].get("data", "")

    return {
        "id": message["id"],
        "subject": subject,
        "date": date,
        "sender": sender,
        "body": body,
    }


def summarize_email(email_content: Dict) -> str:
    """
    Generate summary using an LLM
    You'll want to replace this with actual LLM integration
    """
    # Placeholder for LLM integration
    return "Summary placeholder"


def main():
    st.title("Gmail Star ⭐ Analysis")

    # Initialize Gmail service
    if "gmail_service" not in st.session_state:
        try:
            st.session_state.gmail_service = get_gmail_service()
            st.success("Successfully connected to Gmail!")
        except Exception as e:
            st.error(f"Failed to connect to Gmail: {e}")
            return

    # Fetch emails button
    if st.button("Fetch Starred Emails"):
        with st.spinner("Fetching starred emails..."):
            messages = get_starred_emails(st.session_state.gmail_service)

            if messages:
                # Process emails and store in session state
                emails = []
                progress_bar = st.progress(0)

                for i, message in enumerate(messages):
                    email_details = extract_email_details(
                        st.session_state.gmail_service, message
                    )
                    emails.append(email_details)
                    progress_bar.progress((i + 1) / len(messages))

                st.session_state.emails = emails
                st.success(f"Found {len(emails)} starred emails!")
            else:
                st.warning("No starred emails found.")

    # Display emails if they're in session state
    if "emails" in st.session_state:
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(st.session_state.emails)

        # Basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", len(df))
        with col2:
            st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
        with col3:
            st.metric("Unique Senders", df["sender"].nunique())

        # Email selection and summarization
        st.subheader("Email Browser")
        selected_email = st.selectbox(
            "Select an email to summarize:",
            options=df.index,
            format_func=lambda x: f"{df.iloc[x]['date']} - {df.iloc[x]['subject'][:50]}...",
        )

        if selected_email is not None:
            email = df.iloc[selected_email]

            # Display email details
            with st.expander("Email Details", expanded=True):
                st.write(f"**From:** {email['sender']}")
                st.write(f"**Date:** {email['date']}")
                st.write(f"**Subject:** {email['subject']}")

                # Add summarize button
                if st.button("Summarize This Email"):
                    with st.spinner("Generating summary..."):
                        summary = summarize_email(email)
                        st.write("**Summary:**")
                        st.write(summary)

    if "emails" in st.session_state and "vectorstore" not in st.session_state:
        if st.button("Build Email Knowledge Base"):
            with st.spinner("Building email knowledge base..."):
                df = pd.DataFrame(st.session_state.emails)
                splits = prepare_email_documents(df)
                st.write(f"Extracted {len(splits)} email chunks")
                vector_progress = st.progress(0)
                st.session_state.vectorstore = create_vector_store_with_monitoring(
                    splits, vector_progress
                )
                st.success("Email knowledge base created!")

    if "vectorstore" in st.session_state:
        st.subheader("Email Collection Analysis")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Extract Themes"):
                with st.spinner("Analyzing themes..."):
                    themes = analyze_themes(st.session_state.vectorstore)
                    st.session_state.themes = themes

        with col2:
            if st.button("Generate Overview"):
                with st.spinner("Generating overview..."):
                    overview = generate_overview(st.session_state.vectorstore)
                    st.session_state.overview = overview

        # Display results
        if "themes" in st.session_state:
            with st.expander("Thematic Analysis", expanded=True):
                st.write(st.session_state.themes)

        if "overview" in st.session_state:
            with st.expander("Email Collection Overview", expanded=True):
                st.write(st.session_state.overview)

        # Add search capability
        st.subheader("Search Your Emails")
        search_query = st.text_input("Enter a topic or theme to explore:")
        if search_query:
            results = st.session_state.vectorstore.similarity_search(search_query, k=5)
            for doc in results:
                st.text(doc.page_content[:200] + "...")


if __name__ == "__main__":
    main()
