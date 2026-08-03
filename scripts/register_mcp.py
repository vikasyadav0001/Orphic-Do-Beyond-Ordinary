"""
One-time Dynamic Client Registration for MCP.

Registers Orphic as an OAuth client with the MCP server.
The returned client_id is permanent — store it in .env and use forever.

Usage:
    python scripts/register_mcp.py


"""

import httpx
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import get_settings
env = get_settings()

REGISTRATION_ENDPOINT = "https://cf.mcp.atlassian.com/v1/register"

# Your callback URLs — must match what oauth_engine generates
REDIRECT_URIS = [
    "http://localhost:8000/connections/atlassian/callback",
    # Add production URL if you have one:
    "https://orphic-api.centralindia.cloudapp.azure.com/connections/atlassian/callback",
]

def register():
    """Register Orphic as a public OAuth client with MCP."""
    
    metadata = {
        "client_name": "Orphic",
        "redirect_uris": REDIRECT_URIS,
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",  # public client, PKCE only
    }

    print(f"Registering with: {REGISTRATION_ENDPOINT}")
    print(f"Metadata: {json.dumps(metadata, indent=2)}\n")

    response = httpx.post(
        REGISTRATION_ENDPOINT,
        json=metadata,
        headers={"Content-Type": "application/json"},
    )

    print(f"Status: {response.status_code}")
    
    if response.status_code in (200, 201):
        data = response.json()
        print(f"\n✅ Registration successful!\n")
        print(f"Full response:\n{json.dumps(data, indent=2)}\n")
        
        client_id = data.get("client_id")
        if client_id:
            print(f"{'='*60}")
            print(f"Add this to your .env:")
            print(f"")
            print(f"  MCP_CLIENT_ID={client_id}")
            print(f"{'='*60}")
        else:
            print("⚠️  No client_id in response — check full output above")
    else:
        print(f"\n❌ Registration failed!")
        print(f"Response: {response.text}")


if __name__ == "__main__":
    register()
