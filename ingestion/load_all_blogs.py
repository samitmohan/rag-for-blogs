import logging
import os
from typing import List, Tuple

import requests

logger = logging.getLogger(__name__)

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
    url = f"{GITHUB_API}/repos/{OWNER}/{REPO}/contents/{POSTS_PATH}"
    resp = requests.get(url, headers=_gh_headers())
    resp.raise_for_status()
    return resp.json()


def fetch_post_raw(download_url: str) -> str:
    resp = requests.get(download_url, headers=_gh_headers())
    resp.raise_for_status()
    return resp.text


def fetch_all_posts() -> List[Tuple[str, str]]:
    posts = list_posts()
    out = []
    for p in posts:
        if p["type"] != "file":
            continue
        raw = fetch_post_raw(p["download_url"])
        out.append((p["name"], raw))
    logger.info("Fetched %d posts from GitHub", len(out))
    return out
