import os.path
import pickle
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from pathlib import Path

# If modifying these scopes, delete the file token.pickle
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

creds_path = Path(__file__).parent.parent.parent / "credentials.json"
token_path = Path(__file__).parent.parent.parent / "token.pickle"


def get_gmail_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens
    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    # Create and return the Gmail service
    service = build("gmail", "v1", credentials=creds)
    return service


# Test the connection
if __name__ == "__main__":
    service = get_gmail_service()
    # Get user's email address
    profile = service.users().getProfile(userId="me").execute()
    print(f"Successfully authenticated for: {profile['emailAddress']}")
