from typing import Any, Callable, Dict, List
from pydantic import create_model

def get_dynamic_validators(functions: List[Dict[str, Any]]) -> Dict[str, Callable[..., Any]]:
    validadores = {}
    
    for func in functions:
        nome_funcao = func["name"]
        parametros = func.get("parameters", {})
        
        campos_pydantic = {}
        for nome_param, detalhes in parametros.items():
            tipo_str = detalhes.get("type", "string")
            
            # MAPEAMENTO ESTRITO PARA A MOULINETTE NÃO CHUMBAR!
            if tipo_str == "number":
                tipo_python = float  # Força sempre a ser 3.0, 5.0, etc.
            elif tipo_str == "integer":
                tipo_python = int
            elif tipo_str == "boolean":
                tipo_python = bool
            else:
                tipo_python = str
                
            campos_pydantic[nome_param] = (tipo_python, ...)
            
        ModeloDinamico = create_model(nome_funcao, **campos_pydantic)
        validadores[nome_funcao] = ModeloDinamico
        
    return validadores