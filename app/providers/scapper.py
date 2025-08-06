from langchain_community.document_loaders import AsyncHtmlLoader                    # импорт асинхронного загрузчика HTML-страниц
from langchain_community.document_transformers import Html2TextTransformer, BeautifulSoupTransformer  # импорт трансформеров для чистки и конвертации HTML

FILE_TO_PARSE = "app/data/links.txt"    # путь к файлу, где лежат ссылки (по одной в строке)
DIR_TO_STORE = "app/docs"               # каталог, куда будут сохраняться очищенные тексты

def getLinks2Parse() -> list:       # функция: читает файл и возвращает список URL-ов
    try:
        with open(FILE_TO_PARSE, "r") as f:   # открываем файл со ссылками
            return [link.strip() for link in f.readlines()]  # убираем \n и формируем список строк
    except:                                   # если файла нет или возникла ошибка чтения
        return []                             # просто возвращаем пустой список

def asyncLoader(links):            # функция: загружает ссылки, чистит HTML и пишет в .txt
    loader = AsyncHtmlLoader(links)   # создаём объект асинхронного загрузчика
    docs = loader.load()              # скачиваем страницы, получаем список Document-ов

    # Transform
    bs_transformer = BeautifulSoupTransformer()   # инициализируем трансформер на BeautifulSoup

    for doc in docs:                              # проходим по каждому документу
        doc.page_content = bs_transformer.remove_unwanted_classnames(  # удаляем блоки с ненужными CSS-классами
            doc.page_content,
            ['new-footer', 'main-header',
             'main-top-block', 'callback__form',
             'new-footer-bottom', 'blog-article-share', 'blog-article-slider',
             'blog-article-menu', 'blog__subscribe',
             'main-top-block__info', 'breadcrumbs'])

    html2text = Html2TextTransformer(ignore_links=True, ignore_images=True)  # превращаем HTML в plain-text (без ссылок и картинок)
    docs_transformed = html2text.transform_documents(docs)                  # применяем трансформацию ко всем документам

    for idx, doc in enumerate(docs_transformed):     # сохраняем каждый очищенный документ в отдельный файл
        with open(f"{DIR_TO_STORE}/document_{idx}.txt", "w+", encoding="utf-8") as f:
            f.write(doc.page_content)                # пишем текст
            print(f"File {DIR_TO_STORE}/document_{idx}.txt saved")  # выводим сообщение об успехе

if __name__ == "__main__":   # точка входа, если скрипт запущен напрямую
    ls = getLinks2Parse()    # получаем список ссылок из файла
    asyncLoader(ls)          # запускаем загрузку, очистку и сохранение
