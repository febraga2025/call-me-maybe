import argparse
import os
import json

from .parser import get_function_defini, get_prompts
from .engine import ConstrainedEngine
from llm_sdk import Small_LLM_Model

def main():
    parser = argparse.ArgumentParser(description="Call Me Maybe")
    parser.add_argument('--functions_definition', type=str, default='data/input/functions_definition.json')
    parser.add_argument('--input', type=str, default='data/input/function_calling_tests.json')
    parser.add_argument('--output', type=str, default='data/output/function_calling_results.json')
    args = parser.parse_args()

    functions = get_function_defini(args.functions_definition)
    prompts = get_prompts(args.input)

    if not functions or not prompts:
        print("Erro ao carregar ficheiros.")
        return

    print("A carregar o modelo Qwen3-0.6B...")
    llm = Small_LLM_Model()
    engine = ConstrainedEngine(llm, functions)
    print(f"\nModelo carregado! A processar {len(prompts)} prompts...\n")

    for prompt_text in prompts:
        print(f"\n--- A processar pergunta: '{prompt_text}' ---")
        
        # Contexto direto para evitar alucinações
        contexto_inteligente = f"Function: fn_add_numbers(a: number, b: number)\nUser: {prompt_text}\nJSON Output:"
        
        input_ids = llm.encode(contexto_inteligente).tolist()[0]
        
        gerados_ids = [] 
        max_tokens = 70 
        
        print("A IA está a gerar:\n", end="", flush=True)
        
        texto_impresso = ""
        
        for passo in range(max_tokens):
            logits = llm.get_logits_from_input_ids(input_ids)
            
            # Pegar nos tokens permitidos como um Set para performance
            tokens_permitidos = set(engine.get_allowed_tokens(gerados_ids, prompt_text))
            
            # Aplicar filtro de restrição
            if len(tokens_permitidos) < len(engine.all_token_ids):
                for i in range(len(logits)):
                    if i not in tokens_permitidos:
                        logits[i] = float('-inf')
                    
            token_escolhido = logits.index(max(logits))
            
            input_ids.append(token_escolhido)
            gerados_ids.append(token_escolhido)
            
            texto_completo = llm.decode(gerados_ids)
            texto_novo = texto_completo[len(texto_impresso):]
            print(texto_novo, end="", flush=True)
            texto_impresso = texto_completo
            
            # Condição de paragem automática (Corrigida e indentada!)
            if texto_completo.strip().endswith("}") and texto_completo.count("{") == texto_completo.count("}"):
                print("\n\n--- JSON concluído com sucesso! ---")
                break
            
        print("\n\n--- Fim da geração desta pergunta ---")

if __name__ == "__main__":
    main()