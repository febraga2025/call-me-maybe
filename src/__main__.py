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

# ======================================================================== #
#                   ANSI COLORS (BONUS: VISUALIZATION)                     #
# ======================================================================== #
BG_BLUE = "\033[44m"
TXT_WHITE = "\033[97m"
COLOR_KEY = "\033[96m"
COLOR_VAL = "\033[92m"
COLOR_RESET = "\033[0m"


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments including the visualizer bonus flag."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable real-time colored token generation")
    return parser.parse_args()


def build_neutral_example(functions: List[Dict[str, Any]]) -> str:
    """Builds a completely neutral example to avoid prompting bias."""
    if not functions:
        return ""

    ex_func = functions[0]
    fake_prompt = "Do something with the given values"
    ex_params_str = []

    for key, value in ex_func.get("parameters", {}).items():
        param_type = value.get("type", "string")
        if param_type in ["number", "integer"]:
            val = "42.0"
        elif param_type == "boolean":
            val = "true"
        else:
            val = '"value"'
        ex_params_str.append(f'\n    "{key}": {val}')

    params_joined = ",".join(ex_params_str)
    return (
        f"EXAMPLE:\nUser prompt: {fake_prompt}\nAssistant: {{\n"
        f"  \"prompt\": \"{fake_prompt}\",\n"
        f"  \"name\": \"{ex_func['name']}\",\n"
        f"  \"parameters\": {{{params_joined}\n  }}\n}}\n\n"
    )


def cleanup_output(data: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """The 'Mop': Cleans semantic noise and restores formatting."""
    if "parameters" not in data:
        return data

    for key, val in data["parameters"].items():
        if isinstance(val, str):
            val = val.strip()
            val = val.replace("\\\\", "\\")
            key_low = key.lower()

            if "query" in key_low or "sql" in key_low:
                val = re.split(r"(?i)\s+on\s+", val)[0]
                val = val.strip("'\" ")

            if "db" in key_low or "database" in key_low:
                val = val.replace(" database", "").strip()

            if "path" in key_low or "file" in key_low:
                val = val.split(" with ")[0].strip()
                if (
                    not val.startswith("/")
                    and "\\" not in val
                    and ("/" + val) in prompt
                ):
                    val = "/" + val
                elif val.startswith("home/"):
                    val = "/" + val

            data["parameters"][key] = val
    return data


def process_llm_output(
    texto: str, engine: ConstrainedEngine, prompt_text: str,
    functions: List, validadores: Dict
) -> Dict[str, Any]:
    """Extracts, cleans, formats, and validates the FSM JSON output."""
    texto_seguro = texto.replace("\\", "\\\\").replace('\\\\"', '\\"')

    inicio = texto_seguro.find("{")
    fim = texto_seguro.rfind("}")

    dados: Dict[str, Any] = {
        "prompt": prompt_text,
        "name": (
            engine.current_func["name"] if engine.current_func else "error"
        ),
        "parameters": {}
    }

    if inicio != -1 and fim != -1:
        try:
            json_str = texto_seguro[inicio:fim+1]
            json_str = re.sub(r"[\x00-\x1f\x7f]", "", json_str)
            parsed = json.loads(json_str)
            dados.update(parsed)
        except Exception:
            pass

    nome_funcao = dados.get("name")
    func_def = next((f for f in functions if f["name"] == nome_funcao), None)

    if func_def and "parameters" in dados:
        for k, v in dados["parameters"].items():
            p_type = func_def.get("parameters", {}).get(k, {}).get("type")
            if p_type in ["number", "integer"] and isinstance(v, (int, float,
                                                                  str)):
                try:
                    dados["parameters"][k] = float(v)
                except ValueError:
                    pass

    dados = cleanup_output(dados, prompt_text)

    if nome_funcao in validadores:
        try:
            params_validados = validadores[nome_funcao](**dados.get(
                "parameters", {})).model_dump()
            dados["parameters"] = params_validados
        except Exception:
            pass

    dados["prompt"] = prompt_text
    return dados


def main() -> None:
    args = parse_arguments()
    functions = get_function_defini(args.functions_definition)
    prompts = get_prompts(args.input)
    if not functions or not prompts:
        return

    print("Loading model...")
    llm = Small_LLM_Model()
    engine = ConstrainedEngine(llm, functions)
    validadores = get_dynamic_validators(functions)

    resultados_finais = []

    base_instructions = (
        "Task: Extract exact values into JSON.\n"
        + build_neutral_example(functions)
    )

    for idx, prompt_text in enumerate(prompts, 1):
        engine.reset_prompt(prompt_text)
        gerados_ids: List[int] = []

        prompt_full = (
            base_instructions
            + f"User prompt: {prompt_text}\nAssistant: "
        )
        input_ids = llm.encode(prompt_full).tolist()[0]

        # --- color headear---
        print(f"\n{COLOR_RESET}{'-'*75}")
        print(
            f"{BG_BLUE}{TXT_WHITE} [{idx}/{len(prompts)}] "
            f"PROMPT: {prompt_text[:55]:<55} {COLOR_RESET}"
        )
        print("AI is generating: ", end="", flush=True)

        for _ in range(200):
            tokens_permitidos = engine.get_allowed_tokens(gerados_ids,
                                                          prompt_text)
            if not tokens_permitidos:
                break

            if len(tokens_permitidos) == 1:
                token = tokens_permitidos[0]
            else:
                logits = np.array(
                    llm.get_logits_from_input_ids(input_ids),
                    dtype=np.float32
                )
                mask: np.ndarray = np.ones(len(logits), dtype=bool)
                mask[tokens_permitidos] = False
                logits[mask] = -np.inf
                token = int(np.argmax(logits))

            input_ids.append(token)
            gerados_ids.append(token)

            token_str = llm.decode([token])

            if args.visualize:
                if engine.state == "NEXT_PARAM":
                    print(f"{COLOR_KEY}{token_str}{COLOR_RESET}",
                          end="", flush=True)
                elif engine.state == "VALUE":
                    print(f"{COLOR_VAL}{token_str}{COLOR_RESET}",
                          end="", flush=True)
                else:
                    print(token_str, end="", flush=True)
            else:
                print(token_str, end="", flush=True)

            if engine.state == "COMPLETED":
                break

        print()
        texto = llm.decode(gerados_ids)
        dados_finais = process_llm_output(
            texto, engine, prompt_text, functions, validadores
        )
        resultados_finais.append(dados_finais)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(resultados_finais, f, indent=2, ensure_ascii=False)

    print(f"\n Success! Results saved in: {args.output}")


if __name__ == "__main__":
    main()
