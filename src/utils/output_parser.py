import re
import json

CODE_BLOCK_RE = re.compile(r"```(?:json|sql)?\s*([\s\S]*?)```", re.IGNORECASE)


def extract_code_block(text: str) -> str:
    match = CODE_BLOCK_RE.search(text or "")
    return match.group(1).strip() if match else ""


def parse_json_block(text: str):
    block = extract_code_block(text)
    if not block:
        match = re.search(r'\{[\s\S]*\}', text or "")
        if match:
            block = match.group(0)
        else:
            return None
    try:
        return json.loads(block)
    except Exception:
        return None


def parse_sql_block(text: str) -> str:
    block = extract_code_block(text)
    if not block:
        block = text or ""

    block = block.replace("sql\n", "").replace("```", "").strip()
    upper_block = block.upper()
    forbidden_commands = ['ALTER', 'DELETE', 'DROP', 'INSERT', 'UPDATE', 'TRUNCATE', 'CREATE', 'REPLACE']
    for cmd in forbidden_commands:
        if cmd in upper_block:
            raise ValueError(f"禁止的SQL命令: {cmd}")
    
    select_pos = upper_block.find("SELECT")
    if select_pos == 0:
        return block
    elif select_pos > 0:
        return block[select_pos:]
    else:
        return block


def extract_markdown_block(text: str) -> str:
    return extract_code_block(text) or (text or "").strip()