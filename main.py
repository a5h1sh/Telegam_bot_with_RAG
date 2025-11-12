import logging
import telebot
from config import API_TOKEN, LOG_LEVEL, CHROMA_DIR
from vector_db import init_vector_db
from llm_manager import LLMManager
from file_handler import init_directories
from bot_handlers import setup_bot

logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger.info("Initializing bot...")
    
    # Initialize directories
    init_directories()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Chroma data dir: {CHROMA_DIR}")
    
    # Initialize vector DB
    collection = init_vector_db()
    
    # Initialize LLM manager
    llm_manager = LLMManager()
    
    # Initialize Telegram bot
    bot = telebot.TeleBot(API_TOKEN)
    
    # Set bot commands
    try:
        bot.set_my_commands([
            telebot.types.BotCommand("start", "Start the bot"),
            telebot.types.BotCommand("help", "Show usage instructions"),
            telebot.types.BotCommand("ask", "Ask questions about documents (RAG)"),
            telebot.types.BotCommand("image", "Upload image for description"),
            telebot.types.BotCommand("info", "Bot information"),
        ])
        logger.info("Bot commands registered")
    except Exception as e:
        logger.warning(f"Failed to set bot commands: {e}")

    # Setup message handlers
    setup_bot(bot, collection, llm_manager)
    
    # Start polling
    logger.info("Starting bot (infinity_polling)...")
    try:
        bot.infinity_polling(skip_pending=True, timeout=20)
    except Exception as e:
        logger.error(f"Bot polling failed: {e}")

if __name__ == "__main__":
    main()