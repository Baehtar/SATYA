import os
from enum import Enum


class MessageType(Enum):

    TEXT = "text"
    IMAGE = "image"
    IMAGE_WITH_CAPTION = "image_with_caption"
    VOICE = "voice"
    VIDEO = "video"
    VIDEO_WITH_CAPTION = "video_with_caption"
    UNSUPPORTED = "unsupported"


VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".flv", ".3gp", ".m4v"
}

AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac",
    ".opus", ".wma", ".m4r", ".mpga", ".amr"
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"
}


def classify_message(message):

    # Video or Video Note
    if message.video or message.video_note:
        if getattr(message, "caption", None):
            return MessageType.VIDEO_WITH_CAPTION
        return MessageType.VIDEO

    # Image + caption
    if message.photo and message.caption:
        return MessageType.IMAGE_WITH_CAPTION

    # Image
    if message.photo:
        return MessageType.IMAGE

    # Voice or Audio file
    if message.voice or message.audio:
        return MessageType.VOICE

    # Document file (video, audio, voice, or image sent as document attachment)
    if message.document:
        mime = (message.document.mime_type or "").lower()
        file_name = (message.document.file_name or "").lower()
        ext = os.path.splitext(file_name)[1]

        is_video = (
            mime.startswith("video/")
            or ext in VIDEO_EXTENSIONS
            or any(file_name.endswith(v_ext) for v_ext in VIDEO_EXTENSIONS)
        )
        if is_video:
            if getattr(message, "caption", None):
                return MessageType.VIDEO_WITH_CAPTION
            return MessageType.VIDEO

        is_audio = (
            mime.startswith("audio/")
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