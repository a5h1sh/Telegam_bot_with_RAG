import logging
from pathlib import Path
from datetime import datetime
from config import DOWNLOADS_FOLDER
from file_handler import store_file_to_db

logger = logging.getLogger(__name__)

def generate_filename(file_name, file_type, file_id):
    """Generate a clean filename with timestamp."""
    if file_name:
        return file_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{file_type}_{timestamp}.{_get_extension(file_type)}"

def _get_extension(file_type):
    """Get file extension based on type."""
    extensions = {
        "photo": "jpg",
        "document": "bin",
        "audio": "mp3",
        "video": "mp4"
    }
    return extensions.get(file_type, "bin")

def setup_bot(bot, collection, llm_manager):
    """Setup all Telegram bot message handlers."""
    
    @bot.message_handler(commands=["start"])
    def send_welcome(message):
        welcome_text = (
            "🤖 Welcome to the Document & Image Intelligence Bot!\n\n"
            "Available commands:\n"
            "/help - Show usage instructions\n"
            "/ask <query> - Ask questions about uploaded documents (RAG)\n"
            "/image - Upload an image for description\n\n"
            "You can also upload documents (PDF, DOCX, TXT) directly."
        )
        bot.reply_to(message, welcome_text)

    @bot.message_handler(commands=["help"])
    def send_help(message):
        help_text = (
            "📖 Usage Instructions:\n\n"
            "1️⃣ *Upload Documents*\n"
            "   Send PDF, DOCX, or TXT files directly to store them in the database.\n\n"
            "2️⃣ *Ask Questions (/ask)*\n"
            "   Use: `/ask <your question>`\n"
            "   Example: `/ask What is the invoice total?`\n"
            "   The bot will search stored documents and provide answers.\n\n"
            "3️⃣ *Image Description (/image)*\n"
            "   Use: `/image` then upload a photo\n"
            "   The bot will generate a description of the image.\n\n"
            "📝 Supported formats:\n"
            "   • Documents: PDF, DOCX, TXT\n"
            "   • Images: JPG, PNG\n\n"
            "💡 Tips:\n"
            "   • Upload relevant documents first\n"
            "   • Ask specific questions for better answers\n"
            "   • Image descriptions use AI vision model"
        )
        bot.reply_to(message, help_text, parse_mode="Markdown")

    @bot.message_handler(commands=["info"])
    def send_info(message):
        info_text = (
            "ℹ️ Bot Information:\n\n"
            "This bot provides:\n"
            "✓ Document storage & retrieval (RAG)\n"
            "✓ Question answering from documents\n"
            "✓ AI-powered image description\n\n"
            "Models used:\n"
            "• Text: google/flan-t5-base\n"
            "• Vision: Salesforce BLIP\n"
            "• Vector DB: ChromaDB"
        )
        bot.reply_to(message, info_text)

    @bot.message_handler(commands=["image"])
    def image_command(message):
        bot.reply_to(message, "📸 Please upload an image and I'll describe it for you.")
        bot.register_next_step_handler(message, process_image_upload)

    def process_image_upload(message):
        """Handle image upload after /image command."""
        if message.content_type != "photo":
            bot.reply_to(message, "❌ Please send an image.")
            return
        
        try:
            file_info = bot.get_file(message.photo[-1].file_id)
            caption = message.caption or "uploaded_image"
            file_name = generate_filename(f"{caption}.jpg", "photo", file_info.file_id)
            file_type = "photo"

            file_path = DOWNLOADS_FOLDER / file_name
            downloaded_file = bot.download_file(file_info.file_path)
            
            with open(file_path, "wb") as f:
                f.write(downloaded_file)
            
            logger.info(f"Image downloaded: {file_name}")

            # Generate caption
            bot.reply_to(message, "🔄 Analyzing image...")
            caption_text = llm_manager.caption_image(file_path)
            
            response = f"🖼️ Image Description:\n\n{caption_text}\n\n✅ Image also stored in database for future queries."
            bot.reply_to(message, response)
            
            # Store in DB
            if store_file_to_db(collection, file_info, file_name, file_path, message.from_user.id, file_type=file_type, llm_manager=llm_manager):
                logger.info(f"Image stored: {file_name}")

        except Exception as e:
            logger.exception(f"Error processing image: {e}")
            bot.reply_to(message, f"❌ Error: {str(e)}")

    @bot.message_handler(content_types=["document", "photo", "audio", "video"])
    def handle_file_upload(message):
        try:
            if message.content_type == "document":
                file_info = bot.get_file(message.document.file_id)
                original_name = message.document.file_name or "document"
                file_name = generate_filename(original_name, "document", file_info.file_id)
                file_type = "document"
            elif message.content_type == "photo":
                file_info = bot.get_file(message.photo[-1].file_id)
                caption = message.caption or "photo"
                file_name = generate_filename(f"{caption}.jpg", "photo", file_info.file_id)
                file_type = "photo"
            elif message.content_type == "audio":
                file_info = bot.get_file(message.audio.file_id)
                original_name = message.audio.file_name or "audio"
                file_name = generate_filename(original_name, "audio", file_info.file_id)
                file_type = "audio"
            elif message.content_type == "video":
                file_info = bot.get_file(message.video.file_id)
                original_name = message.video.file_name or "video"
                file_name = generate_filename(original_name, "video", file_info.file_id)
                file_type = "video"
            else:
                bot.reply_to(message, "❌ Unsupported file type")
                return

            file_path = DOWNLOADS_FOLDER / file_name
            downloaded_file = bot.download_file(file_info.file_path)
            
            with open(file_path, "wb") as f:
                f.write(downloaded_file)
            
            logger.info(f"File downloaded: {file_name}")

            if store_file_to_db(collection, file_info, file_name, file_path, message.from_user.id, file_type=file_type, llm_manager=llm_manager):
                bot.reply_to(message, f"✅ '{file_name}' stored in vector database!")
            else:
                bot.reply_to(message, f"✅ '{file_name}' downloaded but not stored in DB")

        except Exception as e:
            logger.exception(f"Error uploading file: {e}")
            bot.reply_to(message, f"❌ Error: {str(e)}")

    @bot.message_handler(commands=["ask"])
    def ask_command(message):
        parts = message.text.split(maxsplit=1)
        if len(parts) < 2 or not parts[1].strip():
            bot.reply_to(message, "Usage: /ask <your question>\nExample: /ask What is the invoice total?")
            return
        question = parts[1].strip()

        if collection is None:
            bot.reply_to(message, "❌ Vector DB not available.")
            return

        try:
            bot.reply_to(message, "🔍 Searching documents...")
            
            results = collection.query(
                query_texts=[question],
                n_results=4,
                include=["documents", "metadatas", "distances"]
            )
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            
            if not docs:
                bot.reply_to(message, "❌ No relevant documents found. Please upload documents first.")
                return

            contexts = [f"{metas[i].get('filename', 'doc') if isinstance(metas[i], dict) else 'doc'}:\n{d[:2000]}" 
                       for i, d in enumerate(docs)]

            bot.reply_to(message, "🤖 Generating answer...")
            answer = llm_manager.answer(question, contexts)
            
            if not answer:
                bot.reply_to(message, "❌ Model returned no answer.")
                return
            
            if len(answer) > 3900:
                answer = answer[:3900] + "\n\n[truncated]"
            
            response = f"💬 Answer:\n\n{answer}"
            bot.reply_to(message, response)

        except Exception as e:
            logger.exception("ask error")
            bot.reply_to(message, f"❌ Error during /ask: {e}")

    @bot.message_handler(func=lambda message: True)
    def echo_all(message):
        bot.reply_to(message, "👋 I didn't understand that. Use /help for available commands.")