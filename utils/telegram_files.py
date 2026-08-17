import os


async def download_photo(message, bot):

    if not message.photo:
        raise ValueError("Message does not contain a photo")

    photo = message.photo[-1]

    telegram_file = await bot.get_file(
        photo.file_id
    )

    os.makedirs("temp", exist_ok=True)

    path = os.path.join(
        "temp",
        f"{photo.file_unique_id}.jpg"
    )

    await telegram_file.download_to_drive(path)

    return path


async def download_voice(message, bot):

    if not message.voice:
        raise ValueError("Message does not contain voice")

    voice = message.voice

    telegram_file = await bot.get_file(
        voice.file_id
    )

    os.makedirs("temp", exist_ok=True)

    path = os.path.join(
        "temp",
        f"{voice.file_unique_id}.ogg"
    )

    await telegram_file.download_to_drive(path)

    return path