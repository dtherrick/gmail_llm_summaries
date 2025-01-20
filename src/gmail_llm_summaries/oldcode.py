def main():
    st.title("Gmail Star ⭐ Summarizer")

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
            # format_func=lambda x: f"{df.iloc[x]['subject'][:50]}...",
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
