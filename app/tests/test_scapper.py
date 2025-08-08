# tests/test_scapper.py
import os
import re

from app.providers import scapper


def test_getLinks2Parse_missing_returns_empty(tmp_path, monkeypatch):
    # FILE_TO_PARSE уже указывает на временный файл (не существует)
    # функция должна вернуть []
    assert scapper.getLinks2Parse() == []


def test_getLinks2Parse_reads_urls(tmp_path, monkeypatch):
    links_file = tmp_path / "links.txt"
    links_file.write_text(
        "https://example.com/a\nhttps://example.com/b\n", encoding="utf-8"
    )

    monkeypatch.setattr(scapper, "FILE_TO_PARSE", str(links_file))

    links = scapper.getLinks2Parse()
    assert links == ["https://example.com/a", "https://example.com/b"]


def test_asyncLoader_writes_plaintext_files(tmp_path, monkeypatch):
    # Подготовим фиктивный список ссылок (FakeAsyncHtmlLoader их игнорит, но требуются по сигнатуре)
    links = ["https://example.com/a", "https://example.com/b"]

    out_dir = tmp_path / "docs"
    monkeypatch.setattr(scapper, "DIR_TO_STORE", str(out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    scapper.asyncLoader(links)

    # Проверяем, что созданы 2 файла
    files = sorted(
        f
        for f in os.listdir(out_dir)
        if f.startswith("document_") and f.endswith(".txt")
    )
    assert len(files) == 2

    # Читаем содержимое и проверяем, что мусорных блоков нет, а полезный текст остался
    content0 = (out_dir / files[0]).read_text(encoding="utf-8")
    content1 = (out_dir / files[1]).read_text(encoding="utf-8")

    # Нежелательные куски удалены (из remove_unwanted_classnames)
    for unwanted in [
        "HEADER TO DROP",
        "MENU TO DROP",
        "BREADCRUMBS TO DROP",
        "FOOTER TO DROP",
    ]:
        assert unwanted not in content0
        assert unwanted not in content1

    # Полезный текст присутствует
    assert "Полезный текст №1" in content0 or "Полезный текст №1" in content1
    assert "Полезный текст №2" in content0 or "Полезный текст №2" in content1

    # Ссылки и картинки убраны (Html2TextTransformer(ignore_links=True, ignore_images=True))
    assert "http://example.com" not in content0 + content1
    assert re.search(r"\bimg\b", content0 + content1, flags=re.IGNORECASE) is None
