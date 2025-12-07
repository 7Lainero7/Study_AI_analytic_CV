# telegram_bot.py
import os
import asyncio
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from gigachat import GigaChat

# Импортируем AgenticRAGHandler из отдельного файла
from agentic_rag import AgenticRAGHandler

# Загружаем .env
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
CHROMA_PATH = "./vectorstore/chroma_db"

# Глобальные объекты
model = None
chroma_client = None
collection = None
giga_chat = None
agent_handler = None  # Для AgenticRAG

async def init_models():
    """Инициализация всех моделей и компонентов"""
    global model, chroma_client, collection, giga_chat, agent_handler
    
    print("🧠 Загрузка модели эмбеддингов...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("📂 Подключение к ChromaDB...")
    chroma_client = chromadb.PersistentClient(
        path=CHROMA_PATH, 
        settings=Settings(allow_reset=False)
    )
    
    # Проверяем существование коллекции
    try:
        collection = chroma_client.get_collection("resumes")
        print(f"✅ Коллекция найдена, {collection.count()} резюме")
    except Exception as e:
        print(f"❌ Коллекция не найдена: {e}")
        raise Exception("Коллекция резюме не найдена. Сначала запустите build_vector_store.py")

    print("💬 Инициализация GigaChat...")
    giga_chat = GigaChat(
        credentials=GIGACHAT_CREDENTIALS,
        verify_ssl_certs=False,
        model="GigaChat:latest",
        scope="GIGACHAT_API_PERS"
    )
    
    # Тестируем подключение к GigaChat
    try:
        test_response = giga_chat.chat("Тест подключения")
        print(f"✅ GigaChat подключен")
    except Exception as e:
        print(f"❌ Ошибка подключения GigaChat: {e}")
        raise

    print("🤖 Инициализация AgenticRAG...")
    agent_handler = AgenticRAGHandler(model, collection, giga_chat)
    
    print("✅ Все компоненты загружены!")
    return True

# Минимальная длина значимого запроса
MIN_QUERY_LENGTH = 3

def is_valid_query(query: str) -> bool:
    """Проверяет, является ли запрос осмысленным для поиска"""
    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        return False
    
    # Проверяем на простые числа или короткие ответы
    if query.strip().isdigit():
        return False
    
    # Проверяем на очевидный мусор
    inappropriate_words = ['жопа', 'хер', 'бля', 'сука', 'пизда', 'ебал']
    query_lower = query.lower()
    for word in inappropriate_words:
        if word in query_lower:
            return False
    
    return True

async def handle_query(user_query: str) -> str:
    """Основная обработка запроса через AgenticRAG"""
    if not agent_handler:
        await init_models()
    
    try:
        print(f"🔍 AgenticRAG обрабатывает запрос: {user_query}")
        result = await agent_handler.process_query(user_query)
        return result
        
    except Exception as e:
        print(f"❌ Ошибка в AgenticRAG: {e}")
        return f"Произошла ошибка при обработке запроса. Попробуйте сформулировать иначе."

# --- Telegram Bot ---
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Клавиатура с примерами
examples_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="React-разработчики в Москве")],
        [KeyboardButton(text="Python с ML и Docker")],
        [KeyboardButton(text="Frontend с Vue.js и TypeScript")],
        [KeyboardButton(text="Backend разработчики от 3 лет")]
    ],
    resize_keyboard=True,
    one_time_keyboard=True
)

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    """Обработка команды /start"""
    await message.answer(
        "👋 Привет! Я — AI-аналитик по резюме с hh.ru.\n\n"
        "Я использую продвинутую систему AgenticRAG для поиска кандидатов:\n"
        "1. 🤖 Анализирую ваш запрос\n"
        "2. 🔍 Ищу подходящих кандидатов в базе\n"
        "3. 📊 Оцениваю релевантность\n"
        "4. 🔗 Предоставляю ссылки на резюме\n\n"
        "📌 **Примеры запросов:**\n"
        "• React-разработчики в Москве\n"
        "• Python с машинным обучением\n"
        "• Frontend с Vue.js и TypeScript\n"
        "• Backend разработчики от 3 лет опыта\n"
        "• Fullstack с React и Node.js",
        reply_markup=examples_keyboard
    )

@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    """Обработка команды /help"""
    await message.answer(
        "📖 **Как работать с ботом:**\n\n"
        "1. Просто напишите запрос на естественном языке\n"
        "2. Будьте конкретны: указывайте технологии, город, опыт\n"
        "3. Примеры хороших запросов:\n"
        "   • 'Найди React-разработчиков'\n"
        "   • 'Кто знает Python и машинное обучение'\n"
        "   • 'Frontend разработчики в Москве с Vue.js'\n"
        "   • 'Разработчики с Docker и Kubernetes'\n\n"
        "4. Избегайте слишком коротких запросов\n"
        "5. Используйте кнопки для быстрых примеров\n\n"
        "🤖 Бот использует AgenticRAG — интеллектуальную систему поиска."
    )

@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    """Показывает статистику базы данных"""
    try:
        if not collection:
            await init_models()
        
        count = collection.count()
        await message.answer(
            f"📊 **Статистика базы резюме:**\n\n"
            f"• Всего резюме: {count}\n"
            f"• Модель эмбеддингов: all-MiniLM-L6-v2\n"
            f"• LLM: GigaChat\n"
            f"• Архитектура: AgenticRAG\n\n"
            f"База обновлена и готова к поиску!"
        )
    except Exception as e:
        await message.answer(f"❌ Не удалось получить статистику: {str(e)}")

@dp.message(lambda message: is_valid_query(message.text))
async def handle_search_query(message: types.Message):
    """Обработка поисковых запросов"""
    user_query = message.text.strip()
    
    try:
        # Отправляем статус обработки
        status_msg = await message.answer("🤖 Анализирую запрос...")
        
        # Обрабатываем запрос через AgenticRAG
        answer = await handle_query(user_query)
        
        # Обрезаем если слишком длинный
        if len(answer) > 4000:
            answer = answer[:4000] + "\n\n... (сообщение обрезано)"
        
        # Удаляем статус и отправляем результат
        await status_msg.delete()
        await message.answer(answer, parse_mode="Markdown")
        
    except Exception as e:
        print(f"❌ Ошибка обработки запроса: {e}")
        await message.answer(
            f"❌ Произошла ошибка при поиске. Попробуйте еще раз или сформулируйте запрос иначе.\n\n"
            f"Ошибка: {str(e)[:200]}"
        )

@dp.message()
async def handle_other_messages(message: types.Message):
    """Обработка коротких/невалидных сообщений"""
    user_text = message.text.strip()
    
    if len(user_text) < MIN_QUERY_LENGTH:
        await message.answer(
            "🤔 Запрос слишком короткий. Пожалуйста, укажите критерии поиска.\n\n"
            "Например:\n"
            "• React-разработчики\n"
            "• Frontend в Москве\n"
            "• Python с ML от 2 лет\n\n"
            "Используйте /help для подсказок."
        )
    else:
        await message.answer(
            "⚠️ Пожалуйста, задайте конкретный запрос для поиска кандидатов.\n\n"
            "Примеры:\n"
            "• 'Найди React-разработчиков'\n"
            "• 'Кто знает Docker и Kubernetes'\n"
            "• 'Data Scientist с Python'\n\n"
            "Используйте /start для просмотра примеров или /help для справки."
        )

async def main():
    """Основная функция запуска бота"""
    # Инициализируем модели перед запуском
    try:
        await init_models()
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        print("⚠️ Проверьте:\n1. Файл .env с токенами\n2. Существование vectorstore\n3. Доступность GigaChat")
        return
    
    print("🚀 Telegram-бот запущен!")
    print("🤖 Используется AgenticRAG архитектура")
    print("📊 База содержит резюме:", collection.count() if collection else "не загружена")
    
    # Запускаем поллинг
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())