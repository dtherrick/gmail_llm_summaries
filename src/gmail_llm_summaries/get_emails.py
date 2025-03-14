from gmail_llm_summaries.auth import get_gmail_service


def get_label_id(service, label_name):
    # List all labels
    results = service.users().labels().list(userId="me").execute()
    labels = results.get("labels", [])
    if not labels:
        label_id = None
    else:
        label_id = [next((item["id"] for item in labels if item["name"] == label_name), None)]
    return label_id


def get_emails_by_label(service, labels=["STARRED"], max_results=500):
    try:
        results = (
            service.users()
            .messages()
            .list(userId="me", labelIds=labels, maxResults=max_results)
            .execute()
        )

        messages = results.get("messages", [])

        if not messages:
            print("No starred messages found.")
            return []

        print(f"Found {len(messages)} messages.")
        return messages

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def get_emails_by_date_range(service, start_date, end_date, max_results=500):
    try:
        query = f"after:{start_date} before:{end_date}"
        results = (
            service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
        )

        messages = results.get("messages", [])

        if not messages:
            print(f"No messages found between {start_date} and {end_date}.")
            return []

        print(f"Found {len(messages)} messages between {start_date} and {end_date}.")
        return messages

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Test it out
if __name__ == "__main__":
    service = get_gmail_service()

    label_name = "forLLM"
    label_id = get_label_id(service, label_name)
    messages = get_emails_by_label(service, labels=label_id)

    # Get details of first message as a sample
    if messages:
        msg = service.users().messages().get(userId="me", id=messages[0]["id"]).execute()
        print("\nSample message headers:")
        for header in msg["payload"]["headers"]:
            if header["name"].lower() in ["subject", "from", "date"]:
                print(f"{header['name']}: {header['value']}")
