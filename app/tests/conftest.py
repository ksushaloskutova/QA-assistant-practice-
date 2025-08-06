import os
import pytest

TEST_DOC_PATH = "app/docs/test_doc.txt"

@pytest.fixture(scope="module", autouse=True)
def create_test_doc():
    os.makedirs(os.path.dirname(TEST_DOC_PATH), exist_ok=True)
    with open(TEST_DOC_PATH, "w", encoding="utf-8") as f:
        f.write("Это тестовый текст.\n" * 10)
    yield
    os.remove(TEST_DOC_PATH)
