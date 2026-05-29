import json
from typing import Any, Dict, List

class ConstrainedEngine:
    def __init__(self, llm: Any, functions: List[Dict[str, Any]]) -> None:
        self.llm = llm
        self.functions = functions
        self.function_map = {f["name"]: f for f in functions}

        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as fh:
            self.vocab = json.load(fh)

        self.all_token_ids = list(self.vocab.values())
        self.decoded_tokens = {tid: self.llm.decode([tid]) for tid in self.all_token_ids}

        self.tokens_numeros = []
        self.tokens_strings = []
        self.tokens_regex = []
        for tid, text in self.decoded_tokens.items():
            t_clean = text.strip()
            if t_clean and all(c in "0123456789.-" for c in t_clean) and ".." not in t_clean:
                self.tokens_numeros.append(tid)
            if not any(ch in text for ch in ["\n", "\r", "Ċ", "$"]):
                if text.strip() != '"':
                    self.tokens_strings.append(tid)
                    token_has_comment = any(mark in text for mark in ["//", "/*", "*/"])
                    token_has_regex_chars = any(
                        ch.isalnum() or ch in ".^$*+?{}[]()|\\/-_"
                        for ch in text
                    )
                    if not token_has_comment and token_has_regex_chars:
                        self.tokens_regex.append(tid)

        self.t_quote = self.llm.encode('"').tolist()[0][0]
        self.t_cb = self.llm.encode("}").tolist()[0][0]
        self.t_sp = self.llm.encode(" ").tolist()[0][0]
        self.final_end = self.llm.encode('\n  }\n}').tolist()[0]

        self.param_key_tokens = {}
        for func in functions:
            fname = func["name"]
            self.param_key_tokens[fname] = []
            for i, pname in enumerate(list(func.get("parameters", {}).keys())):
                sep = f'\n    "{pname}": ' if i == 0 else f',\n    "{pname}": '
                self.param_key_tokens[fname].append(self.llm.encode(sep).tolist()[0])

        self.t_true_seq = self.llm.encode("true").tolist()[0]
        self.t_false_seq = self.llm.encode("false").tolist()[0]

    def reset_prompt(self, prompt_text: str) -> None:
        self.prompt_text = prompt_text
        self.current_func = None
        self.param_names = []
        self.current_param_idx = 0
        self.state = "PREFIX"
        self.key_progress = 0
        self.string_sub_state = 0
        self.current_string_val = ""
        self.current_len = 0
        self.bool_chosen = None
        self.bool_progress = 0
        self.bool_finished = False
        self.end_progress = 0

        pergunta = prompt_text.replace('\\', '\\\\').replace('"', '\\"')
        self.function_prefixes = {}
        for func in self.functions:
            p = f'{{\n  "prompt": "{pergunta}",\n  "name": "{func["name"]}",\n  "parameters": {{'
            self.function_prefixes[func["name"]] = self.llm.encode(p).tolist()[0]

    def get_allowed_tokens(self, gerados_ids: List[int], prompt_text: str) -> List[int]:
        if self.state == "COMPLETED": return [self.t_sp]

        if gerados_ids:
            last = gerados_ids[-1]
            if self.state == "PREFIX":
                for fname, pfix in self.function_prefixes.items():
                    if gerados_ids[-len(pfix):] == pfix or gerados_ids == pfix:
                        self.current_func = self.function_map[fname]
                        self.param_names = list(self.current_func.get("parameters", {}).keys())
                        self.current_param_idx = 0
                        self.state = "NEXT_PARAM" if self.param_names else "END"
                        break
            elif self.state == "NEXT_PARAM":
                keys = self.param_key_tokens[self.current_func["name"]][self.current_param_idx]
                if self.key_progress < len(keys) and last == keys[self.key_progress]:
                    self.key_progress += 1
                    if self.key_progress == len(keys):
                        self.state, self.string_sub_state, self.current_string_val, self.current_len = "VALUE", 0, "", 0
            elif self.state == "VALUE":
                tipo = self.current_func["parameters"][self.param_names[self.current_param_idx]].get("type", "string")
                if tipo not in ["number", "integer", "boolean"]: tipo = "string"

                if tipo == "string":
                    if self.string_sub_state == 0 and last == self.t_quote:
                        self.string_sub_state = 1
                    elif self.string_sub_state == 1:
                        if last == self.t_quote:
                            self.current_param_idx += 1
                            self.state = "NEXT_PARAM" if self.current_param_idx < len(self.param_names) else "END"
                            self.key_progress = 0
                        else:
                            self.current_string_val += self.decoded_tokens.get(last, "")
                            self.current_len += 1
                elif tipo in ["number", "integer"]:
                    is_last = self.current_param_idx == len(self.param_names) - 1
                    target_end = self.final_end[0] if is_last else self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1][0]
                    if last == target_end:
                        if is_last: self.state, self.end_progress = "END", 1
                        else: self.current_param_idx += 1; self.state, self.key_progress = "NEXT_PARAM", 1
                    else: self.current_len += 1
                elif tipo == "boolean":
                    if self.bool_chosen is None:
                        if last == self.t_true_seq[0]: self.bool_chosen, self.bool_progress = "true", 1
                        elif last == self.t_false_seq[0]: self.bool_chosen, self.bool_progress = "false", 1
                    else:
                        if not self.bool_finished:
                            target = self.t_true_seq if self.bool_chosen == "true" else self.t_false_seq
                            if self.bool_progress < len(target) and last == target[self.bool_progress]:
                                self.bool_progress += 1
                                if self.bool_progress == len(target): self.bool_finished = True
                    if self.bool_finished:
                        is_last = self.current_param_idx == len(self.param_names) - 1
                        target_end = self.final_end[0] if is_last else self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1][0]
                        if last == target_end:
                            if is_last: self.state, self.end_progress = "END", 1
                            else: self.current_param_idx += 1; self.state, self.key_progress = "NEXT_PARAM", 1
            elif self.state == "END":
                if self.end_progress < len(self.final_end) and last == self.final_end[self.end_progress]:
                    self.end_progress += 1
                    if self.end_progress == len(self.final_end): self.state = "COMPLETED"

        if self.state == "COMPLETED": return [self.t_sp]

        if self.state == "PREFIX":
            step = len(gerados_ids)
            allowed = {p[step] for p in self.function_prefixes.values() if step < len(p) and p[:step] == gerados_ids}
            return list(allowed) if allowed else [self.t_cb]

        if self.state == "NEXT_PARAM":
            return [self.param_key_tokens[self.current_func["name"]][self.current_param_idx][self.key_progress]]

        if self.state == "VALUE":
            pname = self.param_names[self.current_param_idx]
            tipo = self.current_func["parameters"][pname].get("type", "string")
            if tipo not in ["number", "integer", "boolean"]: tipo = "string"

            if tipo == "string":
                if self.string_sub_state == 0: return [self.t_quote]

                allowed = [self.t_quote]
                
                # A ÚNICA REGRA PARA REGEX (Limpa e com limite de 7 tokens cravados)
                is_regex = "regex" in pname.lower() or "pattern" in pname.lower()

                if is_regex:
                    if self.current_len >= 7: 
                        return [self.t_quote]
                    
                    allowed_chars = set("[]+-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
                    ultimo_token = self.decoded_tokens.get(gerados_ids[-1], "").strip() if gerados_ids else ""

                    for tid in self.tokens_strings:
                        tok_str = self.decoded_tokens[tid]
                        
                        if '"' in tok_str and tok_str != '\\"': 
                            continue
                        
                        if not all(c in allowed_chars for c in tok_str.strip()): 
                            continue
                        
                        # Anti-gaguez para impedir loops
                        if tok_str.strip() == ultimo_token and len(tok_str.strip()) > 0:
                            continue
                            
                        allowed.append(tid)
                    return allowed
                
                # MODO EXTRAÇÃO (Para SQL, Paths, Templates e Replacement)
                # O replacement vai cair automaticamente aqui, copiando a palavra exata da prompt
                if self.current_len >= 35: return allowed
                for tid in self.tokens_strings:
                    tok_str = self.decoded_tokens[tid]
                    if "\n" in tok_str or "\r" in tok_str or "Ċ" in tok_str: continue
                    if '"' in tok_str and tok_str != '\\"': continue
                    
                    cand = self.current_string_val + tok_str
                    cand_clean = cand.replace('\\"', '"').replace('\\\\', '\\')
                    if cand_clean in self.prompt_text:
                        allowed.append(tid)
                return allowed

            is_last = self.current_param_idx == len(self.param_names) - 1
            target_end = self.final_end[0] if is_last else self.param_key_tokens[self.current_func["name"]][self.current_param_idx + 1][0]

            if tipo in ["number", "integer"]:
                if self.current_len >= 12: return [target_end]
                allowed = list(self.tokens_numeros)
                if self.current_len > 0: allowed.append(target_end)
                return allowed

            if tipo == "boolean":
                if self.bool_chosen is None: return [self.t_true_seq[0], self.t_false_seq[0]]
                if not self.bool_finished:
                    target = self.t_true_seq if self.bool_chosen == "true" else self.t_false_seq
                    return [target[self.bool_progress]]
                return [target_end]

        if self.state == "END":
            if self.end_progress < len(self.final_end): return [self.final_end[self.end_progress]]
            return [self.t_sp]

        return [self.t_cb]