import json
import difflib


def load_standard_map():
    try:
        with open("data/env_variables/data_standard.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        data = {}
    return {
        "name_abbreviation": data.get("name_abbreviation", []),
        "device_name": data.get("device_name", []),
        "plant_name": data.get("plant_name", []),
        "outage_type": data.get("outage_type", []),  # 统一用下划线
        "sysName": data.get("sysName", []),
        "sendrecv": data.get("sendrecv", []),
        "voltage_level": data.get("voltage_level", [])

    }

STANDARD_MAP = load_standard_map()

def data_matching(input_name, field_type):
    if not input_name or not STANDARD_MAP:
        return input_name

    standard_list = STANDARD_MAP.get(field_type, [])
    if not standard_list:
        return input_name

    try:
        best_matches = difflib.get_close_matches(
            input_name,
            standard_list,
            n=1,
            cutoff=0.5
        )
        if best_matches:
            return best_matches[0]
    except:
        pass

    return input_name
