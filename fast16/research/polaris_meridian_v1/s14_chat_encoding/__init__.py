"""DeepSeek-V4 固定 revision 的官方消息编码接口。"""

from .encoding_dsv4 import encode_messages, parse_message_from_completion_text

__all__ = ["encode_messages", "parse_message_from_completion_text"]
