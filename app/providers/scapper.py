# app/providers/fetch_and_clean.py
from __future__ import annotations
import asyncio
import hashlib
import os
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import AsyncHtmlLoader
from bs4 import BeautifulSoup
import json

FILE_TO_PARSE = "app/data/links.txt"
DIR_TO_STORE = Path("app/docs")
DIR_TO_STORE.mkdir(parents=True, exist_ok=True)

UNWANTED_SELECTORS = [
    "nav", "header", "footer", "script", "style",
    ".new-footer", ".main-header", ".main-top-block", ".callback__form",
    ".new-footer-bottom", ".blog-article-share", ".blog-article-slider",
    ".blog-article-menu", ".blog__subscribe", ".main-top-block__info",
    ".breadcrumbs", ".social", ".subscribe", ".cookie", ".banner",
]

def read_links() -> List[str]:
    try:
        with open(FILE_TO_PARSE, "r", encoding="utf-8") as f:
            links = [x.strip() for x in f if x.strip()]
        # убираем дубли и якоря
        norm = []
        seen = set()
        for url in links:
            base = url.split("#")[0]
            if base not in seen:
                seen.add(base)
                norm.append(base)
        return norm
    except FileNotFoundError:
        return []

def slugify(url: str) -> str:
    # делаем короткое стабильное имя файла из URL
    safe = re.sub(r"[^a-zA-Z0-9\-]+", "-", url.lower()).strip("-")
    if len(safe) > 80:
        h = hashlib.md5(url.encode("utf-8")).hexdigest()[:8]
        safe = safe[:72] + "-" + h
    return safe or hashlib.md5(url.encode("utf-8")).hexdigest()[:8]

def clean_html_to_text(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")

    # вырезаем мусор
    for sel in UNWANTED_SELECTORS:
        for tag in soup.select(sel):
            tag.decompose()

    # заголовки
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())

    # собираем контент блоками: h1/h2/h3, p, li, th/td
    lines: List[str] = []
    h1 = soup.find("h1")
    if h1:
        lines.append("# " + " ".join(h1.get_text(" ", strip=True).split()))
        lines.append("")

    for tag in soup.find_all(["h2", "h3", "p", "li", "th", "td"]):
        txt = " ".join(tag.get_text(" ", strip=True).split())
        if not txt:
            continue
        if tag.name in ("h2", "h3"):
            level = "##" if tag.name == "h2" else "###"
            lines.append(f"{level} {txt}")
            lines.append("")
        else:
            lines.append(txt)

    text = "\n".join(lines)

    # пост-очистка: убираем повторы и телефон/почту‑спам
    # подряд идущие одинаковые строки → одну
    uniq = []
    prev = None
    for ln in text.splitlines():
        if ln != prev:
            uniq.append(ln)
        prev = ln
    text = "\n".join(uniq)

    # вырежем длинные блоки ссылок
    text = re.sub(r"(https?://\S+)", "", text)
    # схлопнем множественные пустые строки
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return {"title": title, "text": text}

async def fetch_all(links: List[str], concurrency: int = 5):
    # LangChain AsyncHtmlLoader сам делает asyncio внутри,
    # но мы явно ограничим параллельность.
    results = []
    for i in range(0, len(links), 20):
        batch = links[i:i+20]
        loader = AsyncHtmlLoader(
            batch,
            header_template={"User-Agent": "Mozilla/5.0 (RAGbot; +https://itmo.ru)"}
        )
        docs = loader.load()
        results.extend(docs)
    return results

def save_doc(url: str, title: str, text: str, idx: int):
    slug = slugify(url)
    txt_path = DIR_TO_STORE / f"{idx:03d}_{slug}.txt"
    meta_path = DIR_TO_STORE / f"{idx:03d}_{slug}.meta.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"url": url, "title": title, "file": txt_path.name}, f, ensure_ascii=False, indent=2)

    print(f"[SAVED] {txt_path}")

def run():
    links = read_links()
    if not links:
        print("No links in", FILE_TO_PARSE)
        return

    docs = asyncio.run(fetch_all(links))
    seen_hashes = set()
    saved = 0

    for idx, doc in enumerate(docs, start=1):
        url = doc.metadata.get("source") or doc.metadata.get("url") or doc.metadata.get("source_url") or ""
        cleaned = clean_html_to_text(doc.page_content or "")
        title, text = cleaned["title"], cleaned["text"]

        if not text or len(text) < 300:  # слишком короткие — пропускаем
            print(f"[SKIP] too short: {url}")
            continue

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            print(f"[SKIP] duplicate content: {url}")
            continue
        seen_hashes.add(h)

        save_doc(url=url, title=title, text=text, idx=saved + 1)
        saved += 1

    print(f"Done. Saved {saved} docs to {DIR_TO_STORE}")

if __name__ == "__main__":
    run()
