import os


async def download_photo(message, bot):

    if message.photo:
        photo = message.photo[-1]
        file_id = photo.file_id
        unique_id = photo.file_unique_id
        ext = ".jpg"
    elif message.document:
        doc = message.document
        file_id = doc.file_id
        unique_id = doc.file_unique_id
        filename = getattr(doc, "file_name", "") or ""
        ext = os.path.splitext(filename)[1].lower() or ".jpg"
    else:
        raise ValueError("Message does not contain a photo or image document")

    telegram_file = await bot.get_file(file_id)

    os.makedirs("temp", exist_ok=True)

    path = os.path.join(
        "temp",
        f"{unique_id}{ext}"
    )

    await telegram_file.download_to_drive(path)

    return path


async def download_voice(message, bot):

    if message.voice:
        audio_obj = message.voice
        ext = ".ogg"
    elif message.audio:
        audio_obj = message.audio
        filename = getattr(audio_obj, "file_name", "") or ""
        ext = os.path.splitext(filename)[1].lower() or ".mp3"
    elif message.document:
        audio_obj = message.document
        filename = getattr(audio_obj, "file_name", "") or ""
        ext = os.path.splitext(filename)[1].lower() or ".mp3"
    else:
        raise ValueError("Message does not contain voice, audio, or document")

    telegram_file = await bot.get_file(
        audio_obj.file_id
    )

    os.makedirs("temp", exist_ok=True)

    path = os.path.join(
        "temp",
        f"{audio_obj.file_unique_id}{ext}"
    )

    await telegram_file.download_to_drive(path)

    return path