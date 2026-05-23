from pydantic import create_model
from typing import Dict, Any, Type, List

def get_dynamic_validators(functions_definition: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Lê as definições do JSON e cria classes Pydantic dinamicamente.
    
    Garante que o código aceita qualquer função nova na revisão por pares.
    
    Args:
        functions_definition (List[Dict[str, Any]]): Definições das funções.
        
    Returns:
        Dict[str, Any]: Dicionário mapeando o nome da função para o validador.
    """
    type_mapping: Dict[str, Type[Any]] = {
        "number": float,
        "string": str,
        "boolean": bool
    }
    
    validators: Dict[str, Any] = {}
    for func in functions_definition:
        func_name = str(func.get("name", "UnknownFunction"))
        parameters = func.get("parameters", {})
        
        fields: Dict[str, Any] = {}
        for param_name, param_info in parameters.items():
            tipo_json = param_info.get("type", "string")
            tipo_python = type_mapping.get(tipo_json, str)
            # Define o campo como obrigatório (...) e com o tipo correto
            fields[param_name] = (tipo_python, ...)
            
        # Cria a classe Pydantic dinamicamente em tempo de execução
        validators[func_name] = create_model(func_name, **fields)
        
    return validators