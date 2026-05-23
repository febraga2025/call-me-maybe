import json
from typing import Any, List, Dict, Optional

class ConstrainedEngine:
    """
    Motor de descodificação restrita (FSM). 
    Combina alta velocidade com validação estrita para campos Regex.
    """

    def __init__(self, llm: Any, functions: List[Dict[str, Any]]) -> None:
        self.llm = llm
        self.functions = functions
        self.function_map: Dict[str, Dict[str, Any]] = {f["name"]: f for f in functions}

        vocab_path: str = self.llm.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab: Dict[str, int] = json.load(f)

        self.all_token_ids: List[int] = list(self.vocab.values())
        
        self.tokens_numeros: List[int] = []
        self.tokens_strings: List[int] = []
        self.tokens_regex: List[int] = []  # A TUA IDEIA: Máscara especial para Regex

        for token_id in self.all_token_ids:
            token_text = self.llm.decode([token_id])
            texto_limpo = token_text.strip()
            
            # 1. Numéricos
            if texto_limpo in "0123456789.-" or texto_limpo.replace('.', '', 1).replace('-', '', 1).isdigit():
                self.tokens_numeros.append(token_id)
                
            # 2. Strings Seguras
            if '"' not in token_text and '\n' not in token_text and 'Ċ' not in token_text and token_text != "":
                self.tokens_strings.append(token_id)
                
                # 3. Regex Seguro (Exclui os parênteses e o 'OR' lógico que causam os loops)
                if not any(c in token_text for c in "()|"):
                    self.tokens_regex.append(token_id)

        self.token_virgula: int = self.llm.encode(',').tolist()[0][0]
        self.token_fecha_chaves: int = self.llm.encode('}').tolist()[0][0]
        self.token_newline: int = self.llm.encode('\n').tolist()[0][0]
        self.token_aspas: int = self.llm.encode('"').tolist()[0][0]
        
        # O Regresso da Formatação Bonita (Ajuda o Fast-Forwarding a ser mais rápido!)
        self.final_end_tokens: List[int] = self.llm.encode('\n  }\n}').tolist()[0]
        
        self.param_key_tokens: Dict[str, List[List[int]]] = {}
        for func in functions:
            f_name = func["name"]
            self.param_key_tokens[f_name] = []
            p_names = list(func.get("parameters", {}).keys())
            for idx, p_name in enumerate(p_names):
                target_str = f'\n    "{p_name}": ' if idx == 0 else f',\n    "{p_name}": '
                self.param_key_tokens[f_name].append(self.llm.encode(target_str).tolist()[0])
                
        self.token_true_sequences: List[int] = self.llm.encode("true").tolist()[0]
        self.token_false_sequences: List[int] = self.llm.encode("false").tolist()[0]

        self.current_func: Optional[Dict[str, Any]] = None
        self.param_names: List[str] = []
        self.current_param_idx: int = 0
        self.state: str = "PREFIX"
        self.key_progress: int = 0
        self.string_sub_state: int = 0
        self.current_string_len: int = 0
        self.bool_chosen: Optional[str] = None
        self.bool_progress: int = 0
        self.bool_finished: bool = False
        self.end_progress: int = 0
        self.function_prefixes: Dict[str, List[int]] = {}

    def reset_prompt(self, prompt_text: str) -> None:
        self.current_func = None
        self.param_names = []
        self.current_param_idx = 0
        self.state = "PREFIX"
        self.key_progress = 0
        self.string_sub_state = 0
        self.current_string_len = 0
        self.bool_chosen = None
        self.bool_progress = 0
        self.bool_finished = False
        self.end_progress = 0
        
        self.function_prefixes = {}
        pergunta_escapada = prompt_text.replace('"', '\\"')
        for func in self.functions:
            prefix_str = f'{{\n  "prompt": "{pergunta_escapada}",\n  "name": "{func["name"]}",\n  "parameters": {{'
            self.function_prefixes[func["name"]] = self.llm.encode(prefix_str).tolist()[0]

    def get_allowed_tokens(self, gerados_ids: List[int], pergunta_atual: str) -> List[int]:
        if gerados_ids:
            last_token = gerados_ids[-1]
            
            if self.state == "PREFIX":
                for func_name, prefix_tokens in self.function_prefixes.items():
                    if gerados_ids == prefix_tokens:
                        self.current_func = self.function_map[func_name]
                        self.param_names = list(self.current_func.get("parameters", {}).keys())
                        self.current_param_idx = 0
                        if self.param_names:
                            self.state = "NEXT_PARAM"
                            self.key_progress = 0
                        else:
                            self.state = "END"
                            self.end_progress = 0
                        break
                        
            elif self.state == "NEXT_PARAM":
                key_tokens = self.param_key_tokens[self.current_func["name"]][self.current_param_idx]
                if self.key_progress < len(key_tokens) and last_token == key_tokens[self.key_progress]:
                    self.key_progress += 1
                    if self.key_progress == len(key_tokens):
                        self.state = "VALUE"
                        self.string_sub_state = 0
                        self.current_string_len = 0
                        self.bool_chosen = None
                        self.bool_progress = 0
                        self.bool_finished = False
                        
            elif self.state == "VALUE":
                p_name = self.param_names[self.current_param_idx]
                tipo_esperado = self.current_func["parameters"][p_name].get("type", "")
                
                if tipo_esperado == "string":
                    if self.string_sub_state == 0 and last_token == self.token_aspas:
                        self.string_sub_state = 1
                    elif self.string_sub_state == 1:
                        if last_token == self.token_aspas:
                            self.current_param_idx += 1
                            if self.current_param_idx < len(self.param_names):
                                self.state = "NEXT_PARAM"
                                self.key_progress = 0
                            else:
                                self.state = "END"
                                self.end_progress = 0
                        else:
                            self.current_string_len += 1
                            
                elif tipo_esperado == "number":
                    if self.current_param_idx < len(self.param_names) - 1:
                        next_key_tokens = self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1]
                        if last_token == next_key_tokens[0]:
                            self.current_param_idx += 1
                            self.state = "NEXT_PARAM"
                            self.key_progress = 1
                    else:
                        if last_token == self.final_end_tokens[0]:
                            self.state = "END"
                            self.end_progress = 1
                            
                elif tipo_esperado == "boolean":
                    if self.bool_chosen is None:
                        if last_token == self.token_true_sequences[0]:
                            self.bool_chosen = "true"
                            self.bool_progress = 1
                        elif last_token == self.token_false_sequences[0]:
                            self.bool_chosen = "false"
                            self.bool_progress = 1
                        target_seq = self.token_true_sequences if self.bool_chosen == "true" else self.token_false_sequences
                        if self.bool_chosen and self.bool_progress == len(target_seq):
                            self.bool_finished = True
                    else:
                        if not self.bool_finished:
                            target_seq = self.token_true_sequences if self.bool_chosen == "true" else self.token_false_sequences
                            if self.bool_progress < len(target_seq) and last_token == target_seq[self.bool_progress]:
                                            self.bool_progress += 1
                                            if self.bool_progress == len(target_seq):
                                                self.bool_finished = True
                                    
                    if self.bool_finished:
                        if self.current_param_idx < len(self.param_names) - 1:
                            next_key_tokens = self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1]
                            if last_token == next_key_tokens[0]:
                                self.current_param_idx += 1
                                self.state = "NEXT_PARAM"
                                self.key_progress = 1
                        else:
                            if last_token == self.final_end_tokens[0]:
                                self.state = "END"
                                self.end_progress = 1
                                
            elif self.state == "END":
                if self.end_progress < len(self.final_end_tokens) and last_token == self.final_end_tokens[self.end_progress]:
                    self.end_progress += 1
                    if self.end_progress == len(self.final_end_tokens):
                        self.state = "COMPLETED"

        if self.state == "PREFIX":
            allowed = set()
            passo = len(gerados_ids)
            for func_name, prefix_tokens in self.function_prefixes.items():
                if passo < len(prefix_tokens) and prefix_tokens[:passo] == gerados_ids:
                    allowed.add(prefix_tokens[passo])
            return list(allowed) if allowed else self.all_token_ids
            
        elif self.state == "NEXT_PARAM":
            key_tokens = self.param_key_tokens[self.current_func["name"]][self.current_param_idx]
            if self.key_progress < len(key_tokens):
                return [key_tokens[self.key_progress]]
            return self.all_token_ids
            
        elif self.state == "VALUE":
            p_name = self.param_names[self.current_param_idx]
            tipo_esperado = self.current_func["parameters"][p_name].get("type", "")
            
            if tipo_esperado == "string":
                if self.string_sub_state == 0:
                    return [self.token_aspas]
                else:
                    # A TUA IDEIA BRILHANTE: Tratamento Especial para o Regex
                    limit = 45 
                    if p_name == "regex":
                        limit = 8 # Bloqueia aos 8 tokens para impedir alucinações longas
                        if self.current_string_len >= limit:
                            return [self.token_aspas]
                        # Usa a máscara restrita que bloqueia os ( ) |
                        return self.tokens_regex + [self.token_aspas]
                        
                    elif p_name in ["replacement", "name"]:
                        limit = 10
                        
                    if self.current_string_len >= limit:
                        return [self.token_aspas]
                    return self.tokens_strings + [self.token_aspas]
                    
            elif tipo_esperado == "number":
                allowed_tokens = list(self.tokens_numeros)
                if gerados_ids and gerados_ids[-1] in self.tokens_numeros:
                    if self.current_param_idx < len(self.param_names) - 1:
                        next_key_tokens = self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1]
                        allowed_tokens.append(next_key_tokens[0])
                    else:
                        allowed_tokens.append(self.final_end_tokens[0])
                return allowed_tokens
                
            elif tipo_esperado == "boolean":
                if self.bool_chosen is None:
                    return [self.token_true_sequences[0], self.token_false_sequences[0]]
                if not self.bool_finished:
                    target_seq = self.token_true_sequences if self.bool_chosen == "true" else self.token_false_sequences
                    return [target_seq[self.bool_progress]]
                else:
                    if self.current_param_idx < len(self.param_names) - 1:
                        next_key_tokens = self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1]
                        return [next_key_tokens[0]]
                    else:
                        return [self.final_end_tokens[0]]
                        
        elif self.state == "END":
            if self.end_progress < len(self.final_end_tokens):
                return [self.final_end_tokens[self.end_progress]]
            return [self.token_fecha_chaves]
            
        return self.all_token_ids