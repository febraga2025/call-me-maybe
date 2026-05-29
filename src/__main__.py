import argparse
import json
import os
import re
import numpy as np
from typing import List, Dict, Any

from .parser import get_function_defini, get_prompts
from .schemas import get_dynamic_validators
from .engine import ConstrainedEngine
from llm_sdk import Small_LLM_Model

def build_neutral_example(functions: List[Dict[str, Any]]) -> str:
    """Cria um exemplo dinâmico 100% neutro para não viciar a IA."""
    if not functions: return ""
    ex_func = functions[0]
    
    fake_prompt = "Do something with the given values"
    ex_params_str = []
    
    for k, v in ex_func.get("parameters", {}).items():
        t = v.get("type", "string")
        if t in ["number", "integer"]: val = '42.0'
        elif t == "boolean": val = 'true'
        else: val = '"value"'
        ex_params_str.append(f'\n    "{k}": {val}')

    return f"EXAMPLE:\nUser prompt: {fake_prompt}\nAssistant: {{\n  \"prompt\": \"{fake_prompt}\",\n  \"name\": \"{ex_func['name']}\",\n  \"parameters\": {{{','.join(ex_params_str)}\n  }}\n}}\n\n"

def cleanup_output(data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """A Esfregona que corrige o ruído semântico e repõe escapes"""
    if "parameters" not in data: return data
    
    for k, v in data["parameters"].items():
        if isinstance(v, str):
            v = v.strip()
            # Repõe as barras do Windows para o formato original esperado pela Moulinette
            v = v.replace("\\\\", "\\")
            k_low = k.lower()
            
            # Limpa o SQL
            if "query" in k_low or "sql" in k_low:
                v = re.split(r"(?i)\s+on\s+", v)[0]
                v = v.strip("'\" ")
            
            # Limpa DB
            if "db" in k_low or "database" in k_low:
                v = v.replace(" database", "").replace(" database", "").strip()
            
            # Limpa e formata Paths
            if "path" in k_low or "file" in k_low:
                v = v.split(" with ")[0].strip()
                if not v.startswith("/") and not "\\" in v and ("/" + v) in prompt:
                    v = "/" + v
                elif v.startswith("home/"):
                    v = "/" + v
            
            data["parameters"][k] = v
    return data

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--functions_definition', type=str, default='data/input/functions_definition.json')
    parser.add_argument('--input', type=str, default='data/input/function_calling_tests.json')
    parser.add_argument('--output', type=str, default='data/output/function_calling_results.json')
    args = parser.parse_args()

    functions = get_function_defini(args.functions_definition)
    prompts = get_prompts(args.input)
    if not functions or not prompts: return

    print("A carregar o modelo...")
    llm = Small_LLM_Model()
    engine = ConstrainedEngine(llm, functions)
    validadores = get_dynamic_validators(functions)
    
    resultados_finais = []
    base_instructions = "Task: Extract exact values into JSON.\n" + build_neutral_example(functions)

    for idx, prompt_text in enumerate(prompts, 1):
        engine.reset_prompt(prompt_text)
        gerados_ids = []
        input_ids = llm.encode(base_instructions + f"User prompt: {prompt_text}\nAssistant: ").tolist()[0]
        
        print(f"[{idx}/{len(prompts)}] Processando: {prompt_text[:50]}")
        print("A IA está a gerar: ", end="", flush=True)
        
        for _ in range(200):
            tokens_permitidos = engine.get_allowed_tokens(gerados_ids, prompt_text)
            if not tokens_permitidos: break
            
            if len(tokens_permitidos) == 1:
                token = tokens_permitidos[0]
            else:
                logits = np.array(llm.get_logits_from_input_ids(input_ids), dtype=np.float32)
                mask = np.ones(len(logits), dtype=bool)
                mask[tokens_permitidos] = False
                logits[mask] = -np.inf
                token = int(np.argmax(logits))
            
            input_ids.append(token)
            gerados_ids.append(token)
            
            print(llm.decode([token]), end="", flush=True)
            if engine.state == "COMPLETED": break
        
        print()
        texto = llm.decode(gerados_ids)
        
        # 🚀 O FIX DO JSON PARA WINDOWS: Impede o crash do 'Invalid \escape'
        texto_seguro = texto.replace('\\', '\\\\').replace('\\\\"', '\\"')
        
        inicio, fim = texto_seguro.find('{'), texto_seguro.rfind('}')
        dados = {"prompt": prompt_text, "name": engine.current_func["name"] if engine.current_func else "error", "parameters": {}}
        
        if inicio != -1 and fim != -1:
            try:
                # 2. Corrigir JSON mal formado antes de carregar
                json_str = texto_seguro[inicio:fim+1]
                # Remove caracteres de controlo invisíveis que o LLM gosta de enfiar
                json_str = re.sub(r'[\x00-\x1f\x7f]', '', json_str)
                parsed = json.loads(json_str)
                dados.update(parsed)
            except Exception as e:
                print(f"  ⚠️ Aviso JSON: {e}")

        # Conversão de inteiros para floats antes da validação
        nome_funcao = dados.get("name")
        func_def = next((f for f in functions if f["name"] == nome_funcao), None)
        if func_def and "parameters" in dados:
            for k, v in dados["parameters"].items():
                p_type = func_def.get("parameters", {}).get(k, {}).get("type")
                if p_type in ["number", "integer"] and isinstance(v, (int, float, str)):
                    try: dados["parameters"][k] = float(v)
                    except: pass

        # Aplica a limpeza final (A Esfregona)
        dados = cleanup_output(dados, prompt_text)

        # Valida via Pydantic apenas os parâmetros
        if nome_funcao in validadores:
            try:
                params_validados = validadores[nome_funcao](**dados.get("parameters", {})).model_dump()
                dados["parameters"] = params_validados
            except Exception as e:
                pass
        
        dados["prompt"] = prompt_text
        resultados_finais.append(dados)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(resultados_finais, f, indent=2, ensure_ascii=False)
    print(f"\n🎉 Sucesso! Resultados guardados em: {args.output}")

if __name__ == "__main__":
    main()