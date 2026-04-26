from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict


class AddNumbersParams(BaseModel):
    """Parametro para funcao adicao"""
    a: float
    b: float


class GreetParams(BaseModel):
    """Paramentro para funcao de saudacao"""
    name: str = Field(min_length=1, max_length=15)


class ReverseStringParams(BaseModel):
    """Parametro para  a funcao invert string"""
    s: str = Field(min_length=1, max_length=15)


class SquareRootParams(BaseModel):
    a: float = Field(ge=0.0)


class SubstituteRegexParams(BaseModel):
    """Parametro para substituicao com regex"""
    source_string: str = Field(max_lenght=100)
    regex: str = Field(max_length=100)
    replacement: str = Field(max_length=100)


class FunctionCallResult(BaseModel):
    """Resultado final da chamada das funcoes do projeto"""
    prompt: str = Field(min_lenght=1, max_length=500)
    name: str = Field(min_lenght=1, max_length=100)
    parameters: Dict[str, Any]

    @field_validator('name')
    @classmethod
    def validate_function_name(cls, v: str) -> str:
        """Validacao das funcoes permitidas"""
        allowed_functions = ["fn_add_numbers",
                             "fn_greet",
                             "fn_reverse_string",
                             "fn_get_square_root",
                             "fn_substitute_string_with_regex"]
        if v not in allowed_functions:
            raise ValueError(f"Function '{v}' not supported.")
        return v