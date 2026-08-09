"""工具参数 JSON Schema 校验。

按设计文档方向八：LLM 的 tool_calls 参数在进入工具函数前做 Schema 校验，
非法参数（缺 required、类型错误、enum 越界）在函数入口拦截，
返回 schema_error 信号让 LLM 修正参数重试，避免直接 func(**args) 抛异常。

Schema 来源：data/env_variables/tools_config.json 的 parameters_schema 字段
（与 prompt 展示用的 parameters 中文描述分离，一份定义多处消费：
prompt 生成 / 运行时校验 / SFT 训练指令生成）。
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

TOOLS_CONFIG_PATH = "data/env_variables/tools_config.json"


@lru_cache(maxsize=1)
def _load_tools_config() -> dict:
    """加载工具配置（模块级缓存，加工具后重启服务生效）。"""
    try:
        with open(TOOLS_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("[tool_schema] 工具配置加载失败: %s", e)
        return {}


def get_parameters_schema(tool_name: str) -> dict | None:
    """返回指定工具的 parameters_schema；未配置时返回 None。"""
    tool = _load_tools_config().get("tools", {}).get(tool_name, {})
    schema = tool.get("parameters_schema")
    return schema if isinstance(schema, dict) else None


def _check_type(value, expected_type: str) -> bool:
    """类型检查：number 同时接受 int/float（bool 除外）。"""
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    return True  # 未知类型放行


def validate_tool_args(tool_name: str, args: dict) -> tuple[bool, str]:
    """校验工具参数是否符合 parameters_schema。

    Args:
        tool_name: 工具名。
        args: LLM 传入的参数 dict。

    Returns:
        (ok, error_msg)：ok=False 时 error_msg 描述校验失败原因。
        未注册 schema 的工具返回 (True, "")，向后兼容。
    """
    schema = get_parameters_schema(tool_name)
    if not schema:
        return True, ""

    if not isinstance(args, dict):
        return False, "args 必须是 JSON 对象"

    props = schema.get("properties", {})

    # 1. required 缺失检查
    for req in schema.get("required", []):
        if req not in args:
            return False, f"缺少必填参数: {req}"

    # 2. 类型与 enum 检查（仅检查已提供的参数）
    for name, value in args.items():
        prop = props.get(name)
        if not prop:
            continue  # 未知键忽略
        expected = prop.get("type")
        if expected and not _check_type(value, expected):
            return False, (
                f"参数 '{name}' 类型错误: 期望 {expected}, "
                f"实际 {type(value).__name__}"
            )

        enum_vals = prop.get("enum")
        if enum_vals and value not in enum_vals:
            return False, (
                f"参数 '{name}' 取值非法: {value}, 允许: {enum_vals}"
            )

    return True, ""
