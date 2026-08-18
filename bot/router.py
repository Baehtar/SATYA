import os
from enum import Enum


class MessageType(Enum):

    TEXT = "text"
    IMAGE = "image"
    IMAGE_WITH_CAPTION = "image_with_caption"
    VOICE = "voice"
    UNSUPPORTED = "unsupported"


AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac",
    ".opus", ".wma", ".m4r", ".mpeg", ".mp4", ".mpga", ".3gp", ".amr"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
}


def classify_message(message):

    # Image + caption
    if message.photo and message.caption:

        return MessageType.IMAGE_WITH_CAPTION

    # Image
    if message.photo:

        return MessageType.IMAGE

    # Voice or Audio file
    if message.voice or message.audio:

        return MessageType.VOICE

    # Document file (audio, voice, or image sent as document attachment)
    if message.document:
        mime = (message.document.mime_type or "").lower()
        file_name = (message.document.file_name or "").lower()
        ext = os.path.splitext(file_name)[1]

        is_audio = (
            mime.startswith("audio/")
            or mime.startswith("video/")
            or ext in AUDIO_EXTENSIONS
            or any(file_name.endswith(a_ext) for a_ext in AUDIO_EXTENSIONS)
        )
        if is_audio:
            return MessageType.VOICE

        is_image = (
            mime.startswith("image/")
            or ext in IMAGE_EXTENSIONS
            or any(file_name.endswith(i_ext) for i_ext in IMAGE_EXTENSIONS)
        )
        if is_image:
            if message.caption:
                return MessageType.IMAGE_WITH_CAPTION
            return MessageType.IMAGE

    # Text
    if message.text:

        return MessageType.TEXT

    return MessageType.UNSUPPORTED