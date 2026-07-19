"""Fetch recent related emails from Gmail using the Gmail API + OAuth2.

First run: opens a browser window for Google account consent and saves the
token file. Subsequent runs: loads and auto-refreshes the token silently.
"""

import base64
import re
from datetime import datetime, timezone, timedelta

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from .config import Config, GmailConfig

# URLs that are almost never actual article links
_SKIP_URL_PATTERNS = re.compile(
    r"(unsubscribe|optout|opt-out|manage.*preference|tracking|pixel|"
    r"click\.convertkit|email\.mg\.|list-manage|mailchimp|substack\.com/account|"
    r"twitter\.com|facebook\.com|linkedin\.com/company|instagram\.com)",
    re.IGNORECASE,
)


def _get_credentials(gmail_config: GmailConfig) -> Credentials:
    creds = None
    if gmail_config.token_path.exists():
        creds = Credentials.from_authorized_user_file(str(gmail_config.token_path), gmail_config.scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # Prefer client_id / client_secret from config/.env
            if gmail_config.client_id and gmail_config.client_secret:
                client_config = {
                    "installed": {
                        "client_id": gmail_config.client_id,
                        "client_secret": gmail_config.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": ["http://localhost"],
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, gmail_config.scopes)
            elif gmail_config.credentials_path and gmail_config.credentials_path.exists():
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(gmail_config.credentials_path), gmail_config.scopes
                )
            else:
                raise FileNotFoundError(
                    "Gmail credentials not configured.\n"
                    "Option A: set client_id and client_secret\n"
                    f"Option B: save credentials.json to {gmail_config.credentials_path}"
                )
            creds = flow.run_local_server(port=0)
        gmail_config.token_path.parent.mkdir(parents=True, exist_ok=True)
        gmail_config.token_path.write_text(creds.to_json())

    return creds


def _decode_part(part: dict) -> str:
    """Decode a base64url-encoded message part body."""
    data = part.get("body", {}).get("data", "")
    if not data:
        return ""
    return base64.urlsafe_b64decode(data + "==").decode("utf-8", errors="replace")


def _extract_text(payload: dict) -> str:
    """Recursively extract plain text from a message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        return _decode_part(payload)

    if mime == "text/html":
        html = _decode_part(payload)
        return BeautifulSoup(html, "html.parser").get_text(separator="\n")

    if mime.startswith("multipart/"):
        parts = payload.get("parts", [])
        # Prefer plain text part in multipart/alternative
        for part in parts:
            if part.get("mimeType") == "text/plain":
                text = _decode_part(part)
                if text.strip():
                    return text
        # Fallback: recurse into each part
        for part in parts:
            text = _extract_text(part)
            if text.strip():
                return text

    return ""


def _extract_links(text: str) -> list[str]:
    """Extract http(s) URLs from text, filtering out tracking/unsubscribe links."""
    urls = re.findall(r"https?://[^\s\)\]>\"']+", text)
    seen: set[str] = set()
    result = []
    for url in urls:
        url = url.rstrip(".,;)")
        if url in seen:
            continue
        seen.add(url)
        if not _SKIP_URL_PATTERNS.search(url):
            result.append(url)
    return result


def _sender_name(headers: list[dict]) -> str:
    """Extract a clean sender name from message headers."""
    for h in headers:
        if h["name"].lower() == "from":
            # "Sender Name <email@example.com>" → "Sender Name"
            m = re.match(r'^"?([^"<]+?)"?\s*(?:<.*>)?$', h["value"].strip())
            return m.group(1).strip() if m else h["value"].strip()
    return "Unknown"


def fetch_emails(config: Config, hours: int | None = None, as_of: datetime | None = None) -> list[dict]:
    """Return a list of email dicts from the past *hours* hours up to *as_of*.

    Args:
        config: loaded Config, provides gmail config and pipeline defaults.
        hours:  Lookback window in hours (defaults to config.pipeline.lookback_hours).
        as_of:  Upper bound for email delivery time (defaults to now).
                Emails received after this time are excluded.

    Each dict: {title, source, body, links}
    """
    if hours is None:
        hours = config.pipeline.lookback_hours

    creds = _get_credentials(config.gmail)
    service = build("gmail", "v1", credentials=creds)

    reference = as_of or datetime.now(timezone.utc)
    cutoff = reference - timedelta(hours=hours)
    # Gmail date filter is day-granularity; we'll re-filter by exact time below
    after_date = cutoff.strftime("%Y/%m/%d")
    query = f"after:{after_date}"

    results = service.users().messages().list(
        userId="me", q=query, maxResults=100
    ).execute()
    message_refs = results.get("messages", [])

    emails: list[dict] = []
    for ref in message_refs:
        msg = service.users().messages().get(
            userId="me", id=ref["id"], format="full"
        ).execute()

        # Filter by exact timestamp
        ts_ms = int(msg.get("internalDate", 0))
        msg_time = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        if msg_time < cutoff:
            continue
        if as_of and msg_time > as_of:
            continue

        headers = msg["payload"].get("headers", [])
        subject = next(
            (h["value"] for h in headers if h["name"].lower() == "subject"), ""
        )
        source = _sender_name(headers)
        body = _extract_text(msg["payload"])
        links = _extract_links(body)

        emails.append({
            "title": subject,
            "source": source,
            "body": body[:4000],  # cap to avoid huge LLM prompts
            "links": links[:20],
        })

    return emails
