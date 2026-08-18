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


async def download_video(message, bot):

    if message.video:
        vid_obj = message.video
        filename = getattr(vid_obj, "file_name", "") or ""
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
    elif message.video_note:
        vid_obj = message.video_note
        ext = ".mp4"
    elif message.document:
        vid_obj = message.document
        filename = getattr(vid_obj, "file_name", "") or ""
        ext = os.path.splitext(filename)[1].lower() or ".mp4"
    else:
        raise ValueError("Message does not contain video, video_note, or document")

    telegram_file = await bot.get_file(
        vid_obj.file_id
    )

    os.makedirs("temp", exist_ok=True)

    path = os.path.join(
        "temp",
        f"{vid_obj.file_unique_id}{ext}"
    )

    await telegram_file.download_to_drive(path)

    return path