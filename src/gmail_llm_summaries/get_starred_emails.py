from auth import get_gmail_service


def get_starred_emails(service, max_results=100):
    try:
        results = (
            service.users()
            .messages()
            .list(userId="me", labelIds=["STARRED"], maxResults=max_results)
            .execute()
        )

        messages = results.get("messages", [])

        if not messages:
            print("No starred messages found.")
            return []

        print(f"Found {len(messages)} starred messages.")
        return messages

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# Test it out
if __name__ == "__main__":
    service = get_gmail_service()
    messages = get_starred_emails(service)

    # Get details of first message as a sample
    if messages:
        msg = (
            service.users().messages().get(userId="me", id=messages[0]["id"]).execute()
        )
        print("\nSample message headers:")
        for header in msg["payload"]["headers"]:
            if header["name"].lower() in ["subject", "from", "date"]:
                print(f"{header['name']}: {header['value']}")
