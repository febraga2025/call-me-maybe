import json
from typing import List, Any, Dict, Optional


def load_json_file(path: str) -> Optional[Any]:
    """
    Lê um ficheiro JSON de forma robusta e devolve o seu conteúdo.
    
    Args:
        path (str): O caminho para o ficheiro JSON.
        
    Returns:
        Optional[Any]: Os dados do ficheiro JSON ou None em caso de erro.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: Ficheiro não encontrado em {path}")
        return None
    except json.JSONDecodeError:
        print(f"Erro: JSON mal formatado em {path}")
        return None


def get_function_defini(path: str) -> List[Dict[str, Any]]:
    """
    Retorna a lista de definições de funções a partir do ficheiro JSON.
    
    Args:
        path (str): O caminho para o ficheiro JSON de definições.
        
    Returns:
        List[Dict[str, Any]]: Uma lista com os dicionários de cada função.
    """
    data = load_json_file(path)
    if isinstance(data, list):
        return data
    return []


def get_prompts(path: str) -> List[str]:
    """
    Extrai apenas a lista de prompts para processar.
    
    Args:
        path (str): O caminho para o ficheiro JSON de testes.
        
    Returns:
        List[str]: Uma lista contendo as perguntas/prompts.
    """
    data = load_json_file(path)
    if isinstance(data, list):
        return [str(item.get('prompt', ''))
                for item in data if 'prompt' in item]
    return []
