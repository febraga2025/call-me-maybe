import json
from typing import List, Any, Dict


def load_json_file(path: str) -> Any:
    """Le um arquivo Json de forma robusta"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File Not Found in  {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: JSON bad in {path}")
        return None


def get_function_defini(path: str) -> List[Dict[str, Any]]:
    """Return lista de definicoes"""
    data = load_json_file(path)
    return data if data is not None else []


def get_prompts(path: str) -> List[str]:
    """Extra apenas a lista de pronts par aprocessar"""
    data = load_json_file(path)
    if data:
        return [item['prompt'] for item in data]
    return []
