from enum import Enum


class MessageType(Enum):

    TEXT = "text"
    IMAGE = "image"
    IMAGE_WITH_CAPTION = "image_with_caption"
    VOICE = "voice"
    UNSUPPORTED = "unsupported"


def classify_message(message):

    # Image + caption
    if message.photo and message.caption:

        return MessageType.IMAGE_WITH_CAPTION

    # Image
    if message.photo:

        return MessageType.IMAGE

    # Voice
    if message.voice:

        return MessageType.VOICE

    # Text
    if message.text:

        return MessageType.TEXT

    return MessageType.UNSUPPORTED