"""
通用模糊匹配工具

支持多种匹配策略：
- 精确匹配（exact）
- 前缀匹配（prefix）- 用于日期
- 包含匹配（contains）- 用于名称、描述
- 数值模糊匹配（numeric_fuzzy）- 提取数值后比较
"""

import re
import json
from typing import Any, Dict, List, Optional, Tuple
from src.utils.logging_setup import logger

# 加载模糊匹配配置
try:
    with open("data/env_variables/fuzzy_match_config.json", "r", encoding="utf-8") as f:
        _FUZZY_CONFIG = json.load(f)
except Exception as e:
    logger.warning(f"[模糊匹配] 加载配置失败: {e}，使用默认配置")
    _FUZZY_CONFIG = {"fuzzy_fields": {}, "global_rules": {"enable_numeric_extraction": True}}


def extract_numbers(text: str) -> List[float]:
    """从文本中提取所有数值
    
    例如：
    - "±800kV" -> [800.0]
    - "500-1000MWh" -> [500.0, 1000.0]
    - "abc123def456" -> [123.0, 456.0]
    """
    if not isinstance(text, str):
        return []
    
    # 匹配整数和浮点数
    numbers = re.findall(r'-?\d+\.?\d*', text)
    return [float(n) for n in numbers if n]


def numeric_fuzzy_match(db_value: str, user_input: str) -> bool:
    """数值模糊匹配 - 提取数值后比较
    
    例如：
    - db_value="±800kV", user_input="800" -> True
    - db_value="500-1000MWh", user_input="500" -> True
    """
    db_numbers = extract_numbers(db_value)
    input_numbers = extract_numbers(user_input)
    
    if not db_numbers or not input_numbers:
        return False
    
    # 检查是否有任何数值匹配
    for db_num in db_numbers:
        for input_num in input_numbers:
            if db_num == input_num:
                return True
    
    return False


def fuzzy_match(
    db_value: Any,
    user_input: Any,
    match_type: str = "contains",
    field_name: str = "",
    domain: str = ""
) -> bool:
    """通用模糊匹配函数
    
    Args:
        db_value: 数据库中的值
        user_input: 用户输入的值
        match_type: 匹配类型 (exact/prefix/contains/numeric_fuzzy)
        field_name: 字段名（用于日志）
        domain: 业务域（transmission/transaction）
    
    Returns:
        是否匹配
    """
    # 转换为字符串
    db_str = str(db_value).strip()
    input_str = str(user_input).strip()
    
    if not db_str or not input_str:
        return False
    
    # 精确匹配
    if match_type == "exact":
        result = db_str == input_str
        if result:
            logger.debug(f"[模糊匹配] 精确匹配成功: {field_name}={db_str} == {input_str}")
        return result
    
    # 前缀匹配（用于日期）
    if match_type == "prefix":
        result = db_str.startswith(input_str)
        if result:
            logger.debug(f"[模糊匹配] 前缀匹配成功: {field_name}={db_str} 以 {input_str} 开头")
        return result
    
    # 包含匹配（用于名称、描述）
    if match_type == "contains":
        # 先尝试包含匹配
        if input_str in db_str or input_str.lower() in db_str.lower():
            logger.debug(f"[模糊匹配] 包含匹配成功: {field_name}={db_str} 包含 {input_str}")
            return True
        
        # 再尝试数值模糊匹配
        if _FUZZY_CONFIG.get("global_rules", {}).get("enable_numeric_extraction"):
            if numeric_fuzzy_match(db_str, input_str):
                logger.debug(f"[模糊匹配] 数值模糊匹配成功: {field_name}={db_str} 与 {input_str} 数值相同")
                return True
        
        return False
    
    # 数值模糊匹配
    if match_type == "numeric_fuzzy":
        result = numeric_fuzzy_match(db_str, input_str)
        if result:
            logger.debug(f"[模糊匹配] 数值模糊匹配成功: {field_name}={db_str} 与 {input_str} 数值相同")
        return result
    
    # 默认精确匹配
    return db_str == input_str


def get_fuzzy_match_type(field_name: str, domain: str = "") -> str:
    """获取字段的模糊匹配类型
    
    Args:
        field_name: 字段名
        domain: 业务域（transmission/transaction）
    
    Returns:
        匹配类型 (exact/prefix/contains/numeric_fuzzy)
    """
    fuzzy_fields = _FUZZY_CONFIG.get("fuzzy_fields", {})
    
    # 如果指定了业务域，先查找该域的配置
    if domain and domain in fuzzy_fields:
        domain_config = fuzzy_fields[domain]
        if field_name in domain_config:
            field_config = domain_config[field_name]
            if field_config.get("enabled"):
                return field_config.get("match_type", "contains")
    
    # 如果没有指定业务域或该域没有配置，遍历所有域
    for domain_name, domain_config in fuzzy_fields.items():
        if field_name in domain_config:
            field_config = domain_config[field_name]
            if field_config.get("enabled"):
                return field_config.get("match_type", "contains")
    
    # 默认使用精确匹配（未配置的字段不应模糊匹配，避免 ID 类字段被错误归组）
    return "exact"


def is_fuzzy_field(field_name: str, domain: str = "") -> bool:
    """检查字段是否需要模糊匹配
    
    Args:
        field_name: 字段名
        domain: 业务域（transmission/transaction）
    
    Returns:
        是否需要模糊匹配
    """
    fuzzy_fields = _FUZZY_CONFIG.get("fuzzy_fields", {})
    
    # 如果指定了业务域，先查找该域的配置
    if domain and domain in fuzzy_fields:
        domain_config = fuzzy_fields[domain]
        if field_name in domain_config:
            return domain_config[field_name].get("enabled", False)
    
    # 如果没有指定业务域或该域没有配置，遍历所有域
    for domain_name, domain_config in fuzzy_fields.items():
        if field_name in domain_config:
            return domain_config[field_name].get("enabled", False)
    
    return False


def apply_fuzzy_filter(
    data: List[Dict[str, Any]],
    field_name: str,
    user_input: Any,
    domain: str = ""
) -> List[Dict[str, Any]]:
    """应用模糊匹配过滤
    
    这是一个高级函数，结合了模糊匹配配置和过滤逻辑
    
    Args:
        data: 数据列表
        field_name: 字段名
        user_input: 用户输入
        domain: 业务域
    
    Returns:
        过滤后的数据
    """
    if not data or not field_name:
        return data
    
    # 获取匹配类型
    match_type = get_fuzzy_match_type(field_name, domain)
    
    logger.info(f"[模糊过滤] 字段={field_name}, 匹配类型={match_type}, 用户输入={user_input}")
    
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        
        db_value = item.get(field_name)
        if db_value is None:
            continue
        
        if fuzzy_match(db_value, user_input, match_type, field_name, domain):
            result.append(item)
    
    logger.info(f"[模糊过滤] 过滤结果: {len(data)} -> {len(result)} 条记录")
    
    return result

