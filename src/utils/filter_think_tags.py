from collections.abc import Generator, AsyncGenerator
import re

def filter_think_tags(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    """
    过滤流式输出中的 </think> 标签及其内容。
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
                # 查找 </think> 标签
                think_start = buffer.find("</think>")
                
                if think_start == -1:
                    # 没有找到 </think>，但可能有不完整的标签在末尾
                    # 保留可能的不完整标签（最多 6 个字符 "<think"）
                    safe_len = len(buffer) - 6
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    # 找到 </think>，输出之前的内容
                    if think_start > 0:
                        yield buffer[:think_start]
                    buffer = buffer[think_start + 7:]  # 跳过 "</think>"
                    in_think = True
            else:
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


async def filter_think_tags_async(stream: AsyncGenerator[str, None]) -> AsyncGenerator[str, None]:
    """
    异步过滤流式输出中的 </think> 标签及其内容。
    """
    buffer = ""
    in_think = False
    
    async for chunk in stream:
        if not chunk:
            continue
            
        buffer += chunk
        
        # 处理缓冲区中的内容
        while buffer:
            if not in_think:
                # 查找 </think> 标签
                think_start = buffer.find("</think>")
                
                if think_start == -1:
                    # 没有找到 </think>，但可能有不完整的标签在末尾
                    # 保留可能的不完整标签（最多 6 个字符 "<think"）
                    safe_len = len(buffer) - 6
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    # 找到 </think>，输出之前的内容
                    if think_start > 0:
                        yield buffer[:think_start]
                    buffer = buffer[think_start + 7:]  # 跳过 "</think>"
                    in_think = True
            else:
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


def filter_think_tags_v2(stream: Generator[str, None, None]) -> Generator[str, None, None]:
    """
    正确过滤流式输出中的 <think>...</think> 标签及其内容。
    - 遇到 <think> 开始丢弃，直到遇到 </think> 恢复输出。
    - 处理标签可能跨 chunk 的情况。
    """
    buffer = ""
    in_think = False

    for chunk in stream:
        if not chunk:
            continue
        buffer += chunk

        while buffer:
            if not in_think:
                # 查找开始标签 <think>
                start_pos = buffer.find("<think>")
                if start_pos == -1:
                    # 防截断：保留最后 len("<think>")-1 = 6 个字符
                    safe_len = len(buffer) - 6
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    # 输出开始标签之前的内容
                    if start_pos > 0:
                        yield buffer[:start_pos]
                    # 跳过 <think>，进入丢弃模式
                    buffer = buffer[start_pos + 7:]
                    in_think = True
            else:
                # 查找结束标签 </think>
                end_pos = buffer.find("</think>")
                if end_pos == -1:
                    # 防截断：保留最后 len("</think>")-1 = 7 个字符
                    safe_len = len(buffer) - 7
                    if safe_len > 0:
                        buffer = buffer[safe_len:]
                    break
                else:
                    # 丢弃结束标签之前的所有内容，并跳过 </think>
                    buffer = buffer[end_pos + 8:]
                    in_think = False

    # 处理剩余缓冲：仅当不在 think 中时才输出
    if buffer and not in_think:
        yield buffer


async def filter_think_tags_async_v2(
    stream: AsyncGenerator[str, None]
) -> AsyncGenerator[str, None]:
    """
    异步版本，逻辑同 filter_think_tags_v2。
    """
    buffer = ""
    in_think = False

    async for chunk in stream:
        if not chunk:
            continue
        buffer += chunk

        while buffer:
            if not in_think:
                start_pos = buffer.find("<think>")
                if start_pos == -1:
                    safe_len = len(buffer) - 6
                    if safe_len > 0:
                        yield buffer[:safe_len]
                        buffer = buffer[safe_len:]
                    break
                else:
                    if start_pos > 0:
                        yield buffer[:start_pos]
                    buffer = buffer[start_pos + 7:]
                    in_think = True
            else:
                end_pos = buffer.find("</think>")
                if end_pos == -1:
                    safe_len = len(buffer) - 7
                    if safe_len > 0:
                        buffer = buffer[safe_len:]
                    break
                else:
                    buffer = buffer[end_pos + 8:]
                    in_think = False

    if buffer and not in_think:
        yield buffer


def filter_think_tags_simple(text: str) -> str:
    """
    使用正则表达式一次性过滤完整文本中的 <think></think> 标签。适用于非流式场景。
    """
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
