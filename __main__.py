from llm_sdk import Small_LLM_Model
from .parse import get_function_defini, get_prompts
import os

def main():
    llm = Small_LLM_Model()

    path_fns = "data/input/functions_definition.json"
    path_tests = "data/input/function_calling_tests.json"

    functions = get_function_defini(path_fns)
    prompts = get_prompts(path_tests)

    t_open_brace = llm.encode("{").tolist()[0][0]
    t_quote = llm.encode('"').tolist()[0][0]
    t_colon = llm.encode(":").tolist()[0][0]
    t_comma = llm.encode(",").tolist()[0][0]
    t_close_brace = llm.encode("}").tolist()[0][0]

    print(f"Modelo carregado e {len(prompts)} testes prontos!")

if __name__ == "__main__":
    main()