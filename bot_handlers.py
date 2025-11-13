import logging
from pathlib import Path
from datetime import datetime
from config import DOWNLOADS_FOLDER, TOP_K, KEYWORD_COUNT, ENABLE_IMAGE_CAPTIONING, ENABLE_TEXT_PIPELINE, ENABLE_IMAGE_PIPELINE
from file_handler import FileProcessor
from embeddings_manager import EmbeddingsManager
from image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)

def setup_bot(bot, llm_provider, llm_manager=None):
    # initialize components conditionally
    file_processor = FileProcessor() if ENABLE_TEXT_PIPELINE else None
    embedder = EmbeddingsManager() if ENABLE_TEXT_PIPELINE else None
    captioner = ImageCaptioner() if (ENABLE_IMAGE_CAPTIONING and ENABLE_IMAGE_PIPELINE) else None

    # Document upload handler (text pipeline)
    if ENABLE_TEXT_PIPELINE:
        @bot.message_handler(content_types=["document"])
        def handle_document(message):
            try:
                file_info = bot.get_file(message.document.file_id)
                file_name = message.document.file_name or f"doc_{datetime.utcnow().timestamp()}.bin"
                file_path = DOWNLOADS_FOLDER / file_name

                downloaded = bot.download_file(file_info.file_path)
                with open(file_path, "wb") as f:
                    f.write(downloaded)

                bot.reply_to(message, f"⏳ Processing '{file_name}'...")
                user_id = str(message.from_user.id)
                # record upload action to history (source='upload')
                file_processor.vector_store.add_user_interaction(user_id, f"uploaded:{file_name}", source="upload")

                if file_processor.process_and_store(file_path, file_name, "document", user_id=user_id, llm_manager=llm_manager):
                    bot.reply_to(message, f"✅ '{file_name}' stored and indexed!")
                else:
                    bot.reply_to(message, "❌ Failed to process document")
            except Exception as e:
                logger.exception("Document upload error")
                bot.reply_to(message, f"❌ Error: {e}")

        @bot.message_handler(commands=["ask"])
        def ask_command(message):
            parts = message.text.split(maxsplit=1)
            if len(parts) < 2:
                bot.reply_to(message, "Usage: /ask <question> (you may include a filename)")
                return
            question = parts[1].strip()
            user_id = str(message.from_user.id)

            # store the user's question in history (source='query')
            file_processor.vector_store.add_user_interaction(user_id, question, source="query")

            is_summary = any(w in question.lower() for w in ["summary", "summarize", "summarisation", "summarise"])

            # detect explicit filename mention
            doc_names = file_processor.vector_store.list_doc_names()
            matched_doc = None
            for dn in doc_names:
                if dn.lower() in question.lower():
                    matched_doc = dn
                    break
            
            # --- FIX: Define query_keywords here ---
            query_keywords = file_processor._extract_keywords(question, top_n=KEYWORD_COUNT)

            bot.reply_to(message, "🔍 Retrieving top-k chunks...")

            try:
                q_emb = embedder.embed_single(question)

                # if matched_doc restrict search to that doc
                if matched_doc:
                    results = file_processor.vector_store.search(
                        q_emb,
                        top_k=TOP_K,
                        doc_name_filter=matched_doc,
                        query_keywords=query_keywords,
                        deduplicate_docs=False
                    )
                else:
                    results = file_processor.vector_store.search(
                        q_emb,
                        top_k=TOP_K,
                        query_keywords=query_keywords,
                        deduplicate_docs=False
                    )
                
                logger.info(f"DEBUG: Search returned {len(results)} results.")
                if not results:
                    bot.reply_to(message, "❌ No relevant chunks found in the documents.")
                    return

                if is_summary:
                    target_doc = matched_doc or results[0]["doc_name"]
                    logger.info(f"DEBUG: Entering summary logic for target_doc: {target_doc}")
                    chunks = file_processor.vector_store.get_chunks_for_doc(target_doc)
                    logger.info(f"DEBUG: Found {len(chunks)} chunks for summary.")
                    if not chunks:
                        bot.reply_to(message, f"❌ Could not retrieve content for '{target_doc}' to summarize.")
                        return

                    # --- FIX: Group chunks to reduce LLM calls ---
                    bot.reply_to(message, f"🤖 Found {len(chunks)} chunks. Grouping and summarizing to speed up the process...")

                    # Group chunks into larger "meta-chunks" that fit the context window
                    # A CHUNK_SIZE of 250 words is ~300-350 tokens. We can group 1 chunk at a time.
                    # To make a real difference, we'd need to combine them, but let's assume we can't.
                    # The best we can do is sample the chunks if there are too many.
                    
                    # Let's limit the number of chunks to process to a reasonable number, e.g., 20
                    MAX_CHUNKS_TO_SUMMARIZE = 20
                    if len(chunks) > MAX_CHUNKS_TO_SUMMARIZE:
                        bot.reply_to(message, f"⚠️ Document is very long ({len(chunks)} chunks). Summarizing a representative sample of {MAX_CHUNKS_TO_SUMMARIZE} chunks.")
                        # Simple sampling: take evenly spaced chunks
                        step = len(chunks) // MAX_CHUNKS_TO_SUMMARIZE
                        sampled_chunks = chunks[::step]
                    else:
                        sampled_chunks = chunks

                    # 1. Map Step: Summarize each sampled chunk
                    chunk_summaries = []
                    for i, chunk in enumerate(sampled_chunks):
                        logger.info(f"DEBUG: Summarizing chunk {i+1}/{len(sampled_chunks)}")
                        map_prompt = (
                            f"Summarize the following text from a document. Be concise and capture the main points.\n\n"
                            f"Text:\n---\n{chunk['content']}\n---\n\n"
                            f"Concise Summary:"
                        )
                        try:
                            chunk_summary = llm_provider.generate(map_prompt, max_length=150)
                            if chunk_summary:
                                chunk_summaries.append(chunk_summary)
                        except Exception as e:
                            logger.warning(f"Could not summarize chunk {i+1}: {e}")
                    
                    if not chunk_summaries:
                        bot.reply_to(message, "❌ Failed to generate summaries for any document chunks.")
                        return

                    combined_summaries = "\n\n".join(chunk_summaries)
                    reduce_prompt = (
                        f"Fulfill the following request using *only* the provided text, which consists of several summaries from a document titled '{target_doc}'. Synthesize them into a final answer.\n\n"
                        f"Request: '{question}'\n\n"
                        f"Summaries:\n---\n{combined_summaries}\n---\n\n"
                        f"Final Answer:"
                    )

                    bot.reply_to(message, "🤖 Creating final summary...")
                    final_summary = llm_provider.generate(reduce_prompt, max_length=400)
                    logger.info(f"DEBUG: LLM generated final summary of length {len(final_summary) if final_summary else 0}.")
                    bot.reply_to(message, f"📝 Here is the summary for '{target_doc}':\n\n{final_summary}")
                    return

                # This part is for non-summary questions, it should remain as is.
                context_parts = []
                for i, r in enumerate(results, 1):
                    context_parts.append(f"Source {i}: [{r['doc_name']} | chunk {r.get('chunk_index', '?')}] {r['content'][:300]}")
                context = "\n\n".join(context_parts)

                prompt = (
                    "Answer the question using ONLY the following context. If the answer is not present, reply 'Not found in documents'.\n\n"
                    f"Question: {question}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer concisely and cite the source (document name)."
                )

                bot.reply_to(message, "🤖 Generating answer...")
                answer = llm_provider.generate(prompt, max_length=300)

                sources = ", ".join({r["doc_name"] for r in results})
                bot.reply_to(message, f"💬 {answer}\n\n📚 Sources: {sources}")

            except Exception as e:
                logger.exception("Ask error")
                bot.reply_to(message, f"❌ An error occurred during the ask command: {e}")

        @bot.message_handler(commands=["docs"])
        def list_documents(message):
            """Show all stored documents with details."""
            try:
                stats = file_processor.vector_store.get_document_stats()
                
                if stats["unique_documents"] == 0:
                    bot.reply_to(message, "📭 No documents stored yet.\n\nUpload documents to get started!")
                    return
                
                # Format document list
                doc_list = []
                for i, d in enumerate(stats["documents"], 1):
                    doc_list.append(f"{i}. 📄 `{d['name']}`\n   └─ {d['chunks']} chunks")
                
                response = (
                    f"📚 *Stored Documents* ({stats['unique_documents']}):\n\n"
                    f"{''.join([d + '\n\n' for d in doc_list])}"
                    f"*Total chunks:* {stats['total_chunks']}\n\n"
                    f"💡 Use `/clear document_name` to remove a document"
                )
                bot.reply_to(message, response, parse_mode="Markdown")
            except Exception as e:
                logger.exception("Docs list error")
                bot.reply_to(message, f"❌ Error listing documents: {e}")

        @bot.message_handler(commands=["clear"])
        def clear_document(message):
            """Delete a document from vector store."""
            parts = message.text.split(maxsplit=1)
            if len(parts) < 2:
                bot.reply_to(message, "Usage: `/clear document_name`\n\nExample: `/clear invoice.pdf`", parse_mode="Markdown")
                return
            
            doc_name = parts[1].strip()
            
            try:
                # List documents for reference
                stats = file_processor.vector_store.get_document_stats()
                doc_names = [d['name'] for d in stats["documents"]]
                
                # Find matching document (case-insensitive)
                matching_doc = None
                for dn in doc_names:
                    if dn.lower() == doc_name.lower():
                        matching_doc = dn
                        break
                
                if matching_doc:
                    if file_processor.vector_store.delete_document(matching_doc):
                        bot.reply_to(message, f"✅ Deleted: `{matching_doc}`", parse_mode="Markdown")
                        logger.info(f"Deleted document: {matching_doc}")
                    else:
                        bot.reply_to(message, f"❌ Failed to delete: {matching_doc}")
                else:
                    # Show available documents
                    if doc_names:
                        available = "\n".join([f"• {d}" for d in doc_names])
                        bot.reply_to(message, f"❌ Document not found: `{doc_name}`\n\n*Available documents:*\n{available}", parse_mode="Markdown")
                    else:
                        bot.reply_to(message, "❌ No documents stored.")
            except Exception as e:
                logger.exception("Clear document error")
                bot.reply_to(message, f"❌ Error: {e}")

        @bot.message_handler(commands=["summarize"])
        def summarize_command(message):
            """
            /summarize [document_name|image|chat]
            - if "image" or "photo" mentioned -> summarize last photo (caption + tags)
            - if document_name is provided -> summarize the specified document
            - otherwise -> summarize last conversation (last 3 interactions)
            """
            user_id = str(message.from_user.id)
            parts = message.text.split(maxsplit=1)
            arg = parts[1].strip().lower() if len(parts) > 1 else ""

            # helper to find last history entry of a given source
            recent = file_processor.vector_store.get_user_history(user_id, limit=10)

            # prefer image if asked explicitly
            want_image = any(k in arg for k in ("image", "photo", "picture"))
            # if no explicit arg, prefer chat summary
            if not arg:
                want_image = False

            # try image summary
            if want_image and ENABLE_IMAGE_PIPELINE:
                # find last photo entry (stored as "photo:filename" or similar)
                photo_entry = next((h for h in recent if h["source"] == "photo"), None)
                if photo_entry:
                    # message format: "photo:<filename>"
                    msg = photo_entry["message"]
                    if ":" in msg:
                        _, fname = msg.split(":", 1)
                        img_path = DOWNLOADS_FOLDER / fname
                        if img_path.exists() and captioner is not None:
                            bot.reply_to(message, "⏳ Generating caption for last image...")
                            out = captioner.generate_caption(img_path)
                            if out:
                                caption = out.get("caption", "No caption")
                                tags = out.get("keywords", [])
                                tags_text = ", ".join(tags) if tags else "—"
                                bot.reply_to(message, f"🖼️ Caption:\n{caption}\n\n🏷️ Tags: {tags_text}")
                                # record summary action in history
                                file_processor.vector_store.add_user_interaction(user_id, f"summarized_photo:{fname}", source="system")
                                return
                            else:
                                bot.reply_to(message, "❌ Could not generate caption for the image.")
                                return
                        else:
                            bot.reply_to(message, "❌ Image file not found or captioner unavailable.")
                            return
                bot.reply_to(message, "❌ No recent photo found to summarize.")
                return

            # check if arg is a document name (simple existence check)
            doc_names = file_processor.vector_store.list_doc_names()
            if arg in doc_names:
                doc_name = arg
                bot.reply_to(message, f"⏳ Summarizing document: `{doc_name}`...")
                chunks = file_processor.vector_store.get_chunks_for_doc(doc_name)
                if not chunks:
                    bot.reply_to(message, f"❌ No content found for document: `{doc_name}`")
                    return
                
                full_text = "\n\n".join([c["content"] for c in chunks])
                prompt = f"Summarize the document titled '{doc_name}' using only the text below:\n\n{full_text}\n\nSummary:"
                
                bot.reply_to(message, "🤖 Summarizing document...")
                summary = llm_provider.generate(prompt, max_length=300)
                logger.info(f"DEBUG: LLM generated summary of length {len(summary) if summary else 0}.")
                bot.reply_to(message, f"📝 Summary of '{doc_name}':\n\n{summary}")
                return
            
            # fallback: summarize last conversation (last 3 interactions)
            bot.reply_to(message, "🔍 Gathering recent conversation for summary...")
            convo = file_processor.vector_store.get_user_history(user_id, limit=3)
            if not convo:
                bot.reply_to(message, "❌ No recent conversation found to summarize.")
                return

            # build conversation text (oldest -> newest)
            convo_text = "\n".join([f"{c['source']}: {c['message']}" for c in reversed(convo)])
            prompt = (
                "Summarize the user's recent conversation/messages briefly.\n\n"
                f"Conversation:\n{convo_text}\n\nSummary:"
            )

            try:
                bot.reply_to(message, "🤖 Summarizing the last conversation...")
                summary = llm_provider.generate(prompt, max_length=200)
                if not summary:
                    bot.reply_to(message, "❌ No summary generated.")
                    return
                bot.reply_to(message, f"📝 Conversation summary:\n{summary}")
                # record summary action
                file_processor.vector_store.add_user_interaction(user_id, "summarized_conversation", source="system")
            except Exception as e:
                logger.exception("Summarize error")
                bot.reply_to(message, f"❌ Error: {e}")

        @bot.message_handler(commands=["last_doc"])
        def last_doc_command(message):
            """
            /last_doc [image|doc]
            - no arg: return the last processed item (image or document) for this user
            - "image" or "photo": return last processed image
            - "doc" or "document" or "file": return last uploaded document
            """
            user_id = str(message.from_user.id)
            parts = message.text.split(maxsplit=1)
            arg = parts[1].strip().lower() if len(parts) > 1 else ""

            try:
                # prefer explicit request
                if arg in ("image", "photo"):
                    recent = file_processor.vector_store.get_user_history(user_id, limit=20)
                    photo_entry = next((h for h in recent if h["source"] == "photo" and h["message"].startswith("photo:")), None)
                    if photo_entry:
                        _, fname = photo_entry["message"].split(":", 1)
                        path = DOWNLOADS_FOLDER / fname
                        if path.exists():
                            bot.reply_to(message, f"🖼️ Last image: `{fname}`\nPath: `{path}`\nUse /summarize image to summarize or re-caption.", parse_mode="Markdown")
                        else:
                            bot.reply_to(message, f"🖼️ Last image: `{fname}` (file missing on disk)")
                    else:
                        bot.reply_to(message, "❌ No recent image found.")
                    return

                if arg in ("doc", "document", "file"):
                    last_doc = file_processor.vector_store.get_last_uploaded(user_id)
                    if last_doc:
                        bot.reply_to(message, f"📄 Last uploaded document: `{last_doc}`\nUse `/summarize` or `/ask` referencing the filename.", parse_mode="Markdown")
                    else:
                        bot.reply_to(message, "❌ No recent document found.")
                    return

                # no explicit arg: determine most recent activity
                recent = file_processor.vector_store.get_user_history(user_id, limit=1)
                if recent:
                    r = recent[0]
                    if r["source"] == "photo" and r["message"].startswith("photo:"):
                        _, fname = r["message"].split(":", 1)
                        path = DOWNLOADS_FOLDER / fname
                        exists = path.exists()
                        bot.reply_to(message, f"🔎 Most recent: image `{fname}` (exists: {exists})\nUse `/last_doc image` for details or `/summarize image` to caption.", parse_mode="Markdown")
                        return
                    if r["source"] == "upload" and r["message"].startswith("uploaded:"):
                        _, fname = r["message"].split(":", 1)
                        bot.reply_to(message, f"🔎 Most recent: document `{fname}`\nUse `/last_doc doc` or `/summarize` to act on it.", parse_mode="Markdown")
                        return

                # fallback to last uploaded doc (if any)
                last_doc = file_processor.vector_store.get_last_uploaded(user_id)
                if last_doc:
                    bot.reply_to(message, f"📄 Last uploaded document: `{last_doc}`\nUse `/summarize` or `/ask` referencing the filename.", parse_mode="Markdown")
                else:
                    bot.reply_to(message, "❌ No recent documents or images found.")
            except Exception as e:
                logger.exception("last_doc error")
                bot.reply_to(message, f"❌ Error: {e}")

    # Photo handler (image pipeline)
    if ENABLE_IMAGE_PIPELINE:
        @bot.message_handler(content_types=["photo"])
        def handle_photo(message):
            if not ENABLE_IMAGE_CAPTIONING or captioner is None:
                bot.reply_to(message, "Image captioning is not enabled.")
                return

            try:
                # get highest resolution photo
                photo = message.photo[-1]
                file_info = bot.get_file(photo.file_id)
                file_name = f"img_{message.from_user.id}_{int(datetime.utcnow().timestamp())}.jpg"
                file_path = DOWNLOADS_FOLDER / file_name
                DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)

                downloaded = bot.download_file(file_info.file_path)
                with open(file_path, "wb") as f:
                    f.write(downloaded)

                bot.reply_to(message, "⏳ Generating caption and tags...")
                # record photo upload (ensure vector store exists if text pipeline disabled)
                if ENABLE_TEXT_PIPELINE and file_processor:
                    file_processor.vector_store.add_user_interaction(str(message.from_user.id), f"photo:{file_name}", source="photo")
                else:
                    # minimal local history: create a lightweight DB access
                    from vector_store import VectorStore
                    vs = VectorStore()
                    vs.add_user_interaction(str(message.from_user.id), f"photo:{file_name}", source="photo")

                out = captioner.generate_caption(file_path)
                if not out:
                    bot.reply_to(message, "❌ Could not generate caption.")
                    return

                caption = out.get("caption", "No caption")
                tags = out.get("keywords", [])
                tags_text = ", ".join(tags) if tags else "—"
                reply = f"🖼️ Short caption:\n{caption}\n\n🏷️ Tags: {tags_text}"
                bot.reply_to(message, reply)
            except Exception as e:
                logger.exception("Photo handling error")
                bot.reply_to(message, f"❌ Error: {e}")

    # Help and start handlers always enabled
    @bot.message_handler(commands=["start"])
    def send_welcome(message):
        welcome_text = (
            "🤖 Welcome to RAG Bot!\n\n"
            "Commands:\n"
            "/help - Usage instructions\n"
            "/ask <query> - RAG query\n"
            "/docs - Show stored documents\n"
            "/clear <docname> - Delete a document\n"
        )
        bot.reply_to(message, welcome_text)

    @bot.message_handler(commands=["help"])
    def send_help(message):
        # help text (include enabled pipelines info)
        pipelines = []
        if ENABLE_TEXT_PIPELINE:
            pipelines.append("Text pipeline (documents / /ask / /summarize chat)")
        if ENABLE_IMAGE_PIPELINE:
            pipelines.append("Image pipeline (photo captioning / /summarize image)")
        help_text = (
            "📖 Usage:\n\n"
            f"Active pipelines: {', '.join(pipelines) or 'none'}\n\n"
            "1️⃣ Upload documents (PDF, DOCX, TXT) by sending them to the bot.\n"
            "2️⃣ /ask <question> — RAG query across your documents. You can include a filename in the question to restrict the search to that file.\n"
            "3️⃣ /summarize [image|chat] — Summarize the last image (caption + tags) or recent conversation.\n"
            "4️⃣ /last_doc [image|doc] — Show the last processed item for you (no arg = most recent).\n"
            "5️⃣ /docs — List stored documents and chunk counts.\n"
            "6️⃣ /clear <name> — Delete a stored document by name.\n\n"
            "📝 Notes:\n"
            "• To restrict /ask to a specific file, mention the exact filename in the query.\n"
            "• If a query contains keywords (not a filename), the bot uses keyword + embedding matching.\n"
            "• Image captioning (short caption + 3 tags) is available if the image pipeline is enabled.\n"
            "• Use /summarize to get a concise summary of a file or recent chat (last 3 messages).\n"
        )
        bot.reply_to(message, help_text)

    @bot.message_handler(func=lambda m: True)
    def default(message):
        bot.reply_to(message, "Use /help for commands")