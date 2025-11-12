import logging
import telebot
from config import API_TOKEN, LOG_LEVEL
from llm_provider import LLMProvider
from file_handler import init_directories
from bot_handlers import setup_bot

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting RAG Bot...")
    init_directories()
    
    llm_provider = LLMProvider()
    bot = telebot.TeleBot(API_TOKEN)
    
    try:
        bot.set_my_commands([
            telebot.types.BotCommand("start", "Start bot"),
            telebot.types.BotCommand("help", "Show help"),
            telebot.types.BotCommand("ask", "RAG query"),
            telebot.types.BotCommand("docs", "List stored documents"),
            telebot.types.BotCommand("clear", "Delete a document"),
        ])
        logger.info("✅ Bot commands registered")
    except Exception as e:
        logger.warning(f"Failed to set bot commands: {e}")

    setup_bot(bot, llm_provider)
    logger.info("🤖 Bot running...")
    bot.infinity_polling(skip_pending=True, timeout=20)

if __name__ == "__main__":
    main()