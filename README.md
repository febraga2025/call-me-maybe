rascunho readme.

Data Validation & Integrity

Para garantir a confiabilidade do sistema, implementamos uma camada dupla de validação utilizando Pydantic:

1. Input Constraints: Utilizamos Field para impor limites estritos de caracteres (min_length, max_length) e restrições matemáticas (como ge=0.0 para evitar raízes quadradas de números negativos), impedindo que o modelo gere dados sem sentido ou entre em loops de repetição.

2. Function Guard: Implementamos um field_validator que atua como um "filtro de segurança", garantindo que o nome da função gerada pelo LLM pertença estritamente ao catálogo permitido em functions_definition.json.

3. Output Schema (The "Return"): O resultado final é processado por uma classe FunctionCallResult. Isso garante que a saída siga rigorosamente o formato {"prompt": str, "name": str, "parameters": dict}, convertendo automaticamente os valores para os tipos corretos (ex: forçando float para campos numéricos) e removendo qualquer texto extra gerado acidentalmente pelo modelo.

