import logging
import os
import time

import httpx
from dotenv import load_dotenv
from telebot import TeleBot, types
from telebot.util import quick_markup

# Настройка логгирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

# Конфигурация
API_URL = os.getenv('API_URL', 'http://localhost:8000')
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
API_TIMEOUT = 30  # Увеличиваем таймаут для API

if not TELEGRAM_TOKEN:
    raise ValueError("Не указан TELEGRAM_TOKEN в переменных окружения")

# Инициализация бота
bot = TeleBot(TELEGRAM_TOKEN)

# Клиент HTTP с настройками
client = httpx.Client(timeout=API_TIMEOUT)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message: types.Message):
    """Обработчик команд /start и /help"""
    user = message.from_user
    welcome_text = (
        f"Привет, {user.first_name}!\n\n"
        "Я бот-помощник по поступлению в AI Talented Hub ИТМО.\n"
        "Можешь задать мне вопросы о программе, требованиях, сроках подачи и других деталях.\n\n"
        "Примеры вопросов:\n"
        "• Какие документы нужны для поступления?\n"
        "• Какие есть направления подготовки?\n"
        "• Когда крайний срок подачи документов?\n\n"
        "Просто напиши свой вопрос, и я постараюсь помочь!"
    )

    bot.send_message(message.chat.id, welcome_text)


def generate_typing_indicator(chat_id, duration=5):
    """Показывает индикатор набора сообщения в течение указанного времени"""
    end_time = time.time() + duration
    while time.time() < end_time:
        bot.send_chat_action(chat_id, 'typing')
        time.sleep(3)


@bot.message_handler(func=lambda message: True)
def handle_message(message: types.Message):
    """Обработка всех текстовых сообщений"""
    chat_id = message.chat.id
    user_message = message.text.strip()

    if not user_message:
        bot.send_message(chat_id, "Пожалуйста, введите текст вопроса.")
        return

    try:
        # Запускаем индикатор набора в отдельном потоке
        import threading

        typing_thread = threading.Thread(
            target=generate_typing_indicator, args=(chat_id,)
        )
        typing_thread.start()

        # Отправляем запрос в RAG API
        response = client.post(
            f"{API_URL}/chat/{chat_id}",
            json={"question": user_message},
        )

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            bot.send_message(
                chat_id, "Произошла ошибка при обработке запроса. Попробуйте позже."
            )
            logger.error(f"API error: {data['error']}")
            return

        # Отправляем ответ с кнопками для оценки
        markup = quick_markup(
            {
                '👍 Хороший ответ': {'callback_data': 'feedback_good'},
                '👎 Можно лучше': {'callback_data': 'feedback_bad'},
            },
            row_width=2,
        )

        # Разбиваем длинные сообщения на части
        response_text = data["response"]
        if len(response_text) > 4000:
            for i in range(0, len(response_text), 4000):
                bot.send_message(chat_id, response_text[i : i + 4000])
        else:
            bot.send_message(chat_id, response_text, reply_markup=markup)

    except httpx.ReadTimeout:
        bot.send_message(chat_id, "Сервис временно недоступен. Попробуйте позже.")
        logger.error("API request timeout")
    except httpx.RequestError as e:
        bot.send_message(chat_id, "Ошибка соединения с сервисом. Попробуйте позже.")
        logger.error(f"API connection error: {str(e)}")
    except Exception as e:
        bot.send_message(
            chat_id, "Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже."
        )
        logger.exception(f"Error processing message: {e}")
    finally:
        typing_thread.join()  # Завершаем поток с индикатором


@bot.callback_query_handler(func=lambda call: True)
def handle_feedback(call: types.CallbackQuery):
    """Обработка feedback от пользователя"""
    try:
        chat_id = call.message.chat.id
        message_id = call.message.message_id

        if call.data == 'feedback_good':
            feedback_text = "Спасибо за вашу оценку! Рады помочь!"
        elif call.data == 'feedback_bad':
            feedback_text = "Спасибо за обратную связь! Постараемся улучшить ответ."
        else:
            return

        # Удаляем кнопки и отправляем сообщение
        bot.edit_message_reply_markup(chat_id, message_id, reply_markup=None)
        bot.send_message(chat_id, feedback_text)

        # Логируем feedback
        logger.info(f"User {chat_id} feedback: {call.data}")

    except Exception as e:
        logger.error(f"Error handling feedback: {e}")


if __name__ == '__main__':
    logger.info("Starting Telegram bot...")
    try:
        bot.infinity_polling()
    finally:
        client.close()  # Закрываем HTTP-клиент при завершении
