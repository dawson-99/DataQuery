from typing import List, Any
from collections.abc import Generator
from langchain_core.messages import BaseMessage, RemoveMessage
from langchain.agents import AgentState
from langgraph.runtime import Runtime
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from src.utils.logging_setup import logger
from src.config import settings
import re

def trim_messages_by_length(state: AgentState, system_prompt_len: int) -> dict[str, Any] | None:
    """
    根据字符数限制修剪消息历史。
    保留策略：
    1. 系统提示词 (System Prompt) 优先（由外部传入长度占位）。
    2. 最新的消息（通常是 User Query）优先保留。
    3. 历史消息（Memory）保留最近的，直到填满剩余空间。
    """
    messages = state["messages"]
    if not messages:
        return None

    # 剩余可用额度
    available_chars = settings.MAX_CONTEXT_CHARS - system_prompt_len
    
    if available_chars <= 0:
        logger.warning(f"System prompt length ({system_prompt_len}) exceeds limit ({settings.MAX_CONTEXT_CHARS}). Clearing all history.")
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:-1]]}

    # 总是保留最后一条消息 (Current User Input)
    last_msg = messages[-1]
    last_msg_len = len(last_msg.content) if last_msg.content else 0
    if last_msg_len > available_chars:
        # 理论上 AgentsManager 应该处理过，但这里做个兜底
        logger.warning(f"User query length ({last_msg_len}) exceeds available ({available_chars}). Truncating in trimmer.")
        pass

    current_chars = last_msg_len
    kept_messages = [last_msg]
    
    # 倒序遍历历史消息
    # messages[:-1] 是历史
    for msg in reversed(messages[:-1]):
        msg_len = len(msg.content) if msg.content else 0
        if current_chars + msg_len <= available_chars:
            kept_messages.insert(0, msg)
            current_chars += msg_len
        else:
            break
            
    # 如果保留的消息数量等于原数量，说明不需要修剪
    if len(kept_messages) == len(messages):
        return None

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept_messages]}


def filter_think_tags(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    """
    过滤流式输出中的 <think></think> 标签及其内容。
    """
    buffer = ""
    in_think = False
    
    for chunk in stream:
        if not chunk:
            continue
            
        buffer += chunk
        
        # 处理缓冲区中的内容
        while buffer:
            if not in_think:
                # 查找 <think> 标签
                think_start = buffer.find("<think>")
                
                if think_start == -1:
                    # 没有找到 <think>，但可能有不完整的标签在末尾
                    # 保留可能的不完整标签（最多 6 个字符 "<think"）
                    safe_len = len(buffer) - 6
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    # 找到 <think>，输出之前的内容
                    if think_start > 0:
                        yield buffer[:think_start]
                    buffer = buffer[think_start + 7:]  # 跳过 "<think>"
                    in_think = True
            else:
                # 在 think 标签内，查找 </think>
                think_end = buffer.find("</think>")
                
                if think_end == -1:
                    safe_len = len(buffer) - 7
                    if safe_len > 0:
                        buffer = buffer[safe_len:]
                    break
                else:
                    buffer = buffer[think_end + 8:]  # 跳过 "</think>"
                    in_think = False
    
    # 处理剩余的缓冲区
    if buffer and not in_think:
        yield buffer


def filter_think_tags_simple(text: str) -> str:
    """
    使用正则表达式一次性过滤完整文本中的 <think></think> 标签。适用于非流式场景。
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
