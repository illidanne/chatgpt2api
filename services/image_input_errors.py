"""Recognize explicit image-tool input failures without treating them as timeouts."""

class ImageInputError(RuntimeError):
    def __init__(self, message: str, conversation_id: str = "") -> None:
        super().__init__(message)
        self.conversation_id = conversation_id


def find_image_input_error(conversation: dict) -> str:
    # Restrict to actual tool outputs; quoted source/user/assistant text is not an error.
    for node in (conversation.get("mapping") or {}).values():
        message = (node or {}).get("message") or {}
        if (message.get("author") or {}).get("role") != "tool":
            continue
        content = message.get("content") or {}
        if content.get("content_type") != "text":
            continue
        text = " ".join(part for part in content.get("parts", []) if isinstance(part, str))
        normalized = text.lower()
        if "no usable image target" in normalized and "edit or restore" in normalized:
            return text
    return ""
