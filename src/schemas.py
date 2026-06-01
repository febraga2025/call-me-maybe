from typing import Any, Dict, List, Type, cast
from pydantic import create_model


def get_dynamic_validators(
    functions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build dynamic Pydantic models for each function definition.

    Returns a mapping from function name to the generated model class.
    """
    validadores: Dict[str, Any] = {}

    for func in functions:
        nome_funcao = func["name"]
        parametros = func.get("parameters", {})

        campos_pydantic = {}
        for nome_param, detalhes in parametros.items():
            tipo_str = detalhes.get("type", "string")

            # strict map
            tipo_python: Type[Any]
            if tipo_str == "number":
                tipo_python = float
            elif tipo_str == "integer":
                tipo_python = int
            elif tipo_str == "boolean":
                tipo_python = bool
            else:
                tipo_python = str

            campos_pydantic[nome_param] = (tipo_python, ...)

        # create_model has complex overloads; ignore mypy's
        # overload resolution here
        ModeloDinamico: Any = create_model(
            nome_funcao,
            **cast(Any, campos_pydantic),
        )
        validadores[nome_funcao] = ModeloDinamico

    return validadores
