import argparse
import os
import json
import numpy as np
from typing import List, Dict, Any

from .parser import get_function_defini, get_prompts
from .engine import ConstrainedEngine
from .schemas import get_dynamic_validators
from llm_sdk import Small_LLM_Model

def main() -> None:
    """Orquestra a execução usando constrained decoding com FSM e buffers incrementais."""
    parser = argparse.ArgumentParser(description="Call Me Maybe")
    parser.add_argument('--functions_definition', type=str, default='data/input/functions_definition.json')
    parser.add_argument('--input', type=str, default='data/input/function_calling_tests.json')
    parser.add_argument('--output', type=str, default='data/output/function_calling_results.json')
    args = parser.parse_args()

    functions = get_function_defini(args.functions_definition)
    prompts = get_prompts(args.input)

    if not functions or not prompts:
        print("Erro: Falha ao carregar funções ou prompts.")
        return

    try:
        print("A carregar o modelo Qwen3-0.6B...")
        llm = Small_LLM_Model()
    except Exception as e:
        print(f"Erro fatal ao carregar o modelo LLM: {e}")
        return
        
    engine = ConstrainedEngine(llm, functions)
    validadores = get_dynamic_validators(functions)
    
    print(f"\nModelo carregado! A processar {len(prompts)} prompts...\n")
    resultados_finais: List[Dict[str, Any]] = []

    # 🚀 OTIMIZAÇÃO DE PROMPT (Few-Shot Prompting)
    # Modelos pequenos precisam de exemplos para saber como preencher os dados,
    # evitando que tentem criar regexes gigantes ou repitam \n infinitamente.
    base_instructions = "Task: Select the EXACT tool that matches the user's intent.\nAvailable tools:\n"
    for f in functions:
        clean_tool = {
            "name": f.get("name"),
            "parameters": {k: v.get("type", "") for k, v in f.get("parameters", {}).items()}
        }
        base_instructions += json.dumps(clean_tool, separators=(',', ':')) + "\n"
        
    base_instructions += (
        "\nCRITICAL RULES:\n"
        "1. Output ONLY a valid JSON calling the correct function.\n"
        "2. Preserve exact punctuation.\n"
        "3. For regex parameters: ALWAYS use general character class. Vowels: [aeiouAEIOU], Numbers: [0-9]+\n\n"
        "EXAMPLES:\n"
        "User prompt: Replace all digits in 'abc123' with X\n"
        'Assistant: {"name":"<tool>","parameters":{"<p1>":"abc123","regex":"[0-9]+","replacement":"X"}}\n\n'
        "User prompt: Substitute the word 'foo' with 'bar' in 'foo and foo'\n"
        'Assistant: {"name":"<tool>","parameters":{"<p1>":"foo and foo","regex":"foo","replacement":"bar"}}\n\n'
    )

    for idx, prompt_text in enumerate(prompts, 1):
        print(f"[{idx}/{len(prompts)}] A processar: '{prompt_text}'")
        
        # Constrói o contexto final com as instruções perfeitas e a pergunta
        contexto_inteligente = base_instructions + f"User prompt: {prompt_text}\nAssistant: "

        input_ids_list: List[int] = llm.encode(contexto_inteligente).tolist()[0]
        gerados_ids: List[int] = [] 
        max_tokens = 200 
        
        engine.reset_prompt(prompt_text)
        texto_total = ""
        
        print("A IA está a gerar: ", end="", flush=True)
        
        for _ in range(max_tokens):
            try:
                tokens_permitidos = engine.get_allowed_tokens(gerados_ids, prompt_text)
                
                # FAST-FORWARDING: A magia que poupa 50% do tempo de processamento!
                if tokens_permitidos and len(tokens_permitidos) == 1:
                    token_escolhido = tokens_permitidos[0]
                else:
                    logits = llm.get_logits_from_input_ids(input_ids_list)
                    logits_np = np.array(logits, dtype=np.float32)
                    
                    if tokens_permitidos and len(tokens_permitidos) < len(engine.all_token_ids):
                        mask = np.ones(len(logits), dtype=bool)
                        mask[tokens_permitidos] = False
                        logits_np[mask] = -np.inf
                        
                    token_escolhido = int(np.argmax(logits_np))
                
                input_ids_list.append(token_escolhido)
                gerados_ids.append(token_escolhido)
                
                fragmento = llm.decode([token_escolhido])
                texto_total += fragmento
                print(fragmento, end="", flush=True)
                
                if engine.state == "COMPLETED":
                    break
            except Exception as e:
                print(f"\nErro durante a geração de tokens: {e}")
                break
        
        print() 
        
        try:
            texto_final = llm.decode(gerados_ids)
            dados_json = json.loads(texto_final)
            dados_json["prompt"] = prompt_text
            
            nome_funcao = dados_json.get("name")
            if nome_funcao in validadores:
                validador = validadores[nome_funcao]
                validador(**dados_json.get("parameters", {}))
                
            resultados_finais.append(dados_json)
            
        except json.JSONDecodeError:
            print("  ⚠️ Aviso: Erro ao fazer parse do JSON.")
        except Exception as e:
            print(f"  ⚠️ Aviso: Erro Pydantic: {e}")

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(resultados_finais, f, indent=2, ensure_ascii=False)
        print(f"\n🎉 Sucesso! Resultados guardados em: {args.output}")
    except IOError as e:
        print(f"\nErro fatal ao guardar o ficheiro de resultados: {e}")

if __name__ == "__main__":
    main()