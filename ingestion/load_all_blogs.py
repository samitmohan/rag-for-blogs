# ingestion/load_all_blogs.py
import os
import requests
from typing import List, Tuple

GITHUB_API = "https://api.github.com"
OWNER = "samitmohan"
REPO = "samitmohan.github.io"
POSTS_PATH = "_posts"

def _gh_headers():
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def list_posts() -> List[dict]:
    """
    List files in the _posts folder using GitHub API.
    Returns list of file metadata (name, path, download_url).
    """
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{POSTS_PATH}"
    resp = requests.get(url, headers=_gh_headers())
    resp.raise_for_status()
    return resp.json()  # each item: {name,path,sha,download_url,...}

def fetch_post_raw(download_url: str) -> str:
    resp = requests.get(download_url, headers=_gh_headers())
    resp.raise_for_status()
    return resp.text

def fetch_all_posts() -> List[Tuple[str,str]]:
    """
    Returns list of (filename, raw_text) tuples
    """
    posts = list_posts()
    out = []
    for p in posts:
        if p["type"] != "file":
            continue
        raw = fetch_post_raw(p["download_url"])
        out.append((p["name"], raw))
    return out

if __name__ == "__main__":
    posts = fetch_all_posts()
    print(f"Found {len(posts)} posts")
    for name, txt in posts:
        print(name, len(txt))
