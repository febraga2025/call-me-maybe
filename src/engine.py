import json
from typing import Any, Dict, List, Optional


class ConstrainedEngine:
    def __init__(self, llm: Any, functions: List[Dict[str, Any]]) -> None:
        self.llm = llm
        self.functions = functions
        self.function_map = {f["name"]: f for f in functions}
        # typed instance attributes (initialized later or here with defaults)
        self.vocab: Dict[str, int] = {}
        self.all_token_ids: List[int] = []
        self.decoded_tokens: Dict[int, str] = {}

        self.tokens_numbers: List[int] = []
        self.tokens_strings: List[int] = []
        self.tokens_regex: List[int] = []

        self.t_quote: int = 0
        self.t_cb: int = 0
        self.t_sp: int = 0
        self.final_end: List[int] = []

        self.param_key_tokens: Dict[str, List[List[int]]] = {}
        self.t_true_seq: List[int] = []
        self.t_false_seq: List[int] = []

        self.prompt_text: str = ""
        self.current_func: Optional[Dict[str, Any]] = None
        self.param_names: List[str] = []
        self.current_param_idx: int = 0
        self.state: str = "PREFIX"
        self.key_progress: int = 0
        self.string_sub_state: int = 0
        self.current_string_val: str = ""
        self.current_len: int = 0
        self.bool_chosen: Optional[str] = None
        self.bool_progress: int = 0
        self.bool_finished: bool = False
        self.end_progress: int = 0

        self.function_prefixes: Dict[str, List[int]] = {}

        # load and prepare token maps
        self._load_vocabulary()
        self._categorize_tokens()
        self._setup_control_tokens()

    def _load_vocabulary(self) -> None:
        """Loads the vocabulary JSON and initializes the token dictionary."""
        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, "r", encoding="utf-8") as fh:
            self.vocab = json.load(fh)

        self.all_token_ids = list(self.vocab.values())
        self.decoded_tokens = {
            tid: self.llm.decode([tid]) for tid in self.all_token_ids
        }

    def _categorize_tokens(self) -> None:
        """Filters tokens into specific lists (numbers, strings)."""
        # lists already initialized in __init__, just reuse/clear
        self.tokens_numbers.clear()
        self.tokens_strings.clear()
        self.tokens_regex.clear()

        for tid, text in self.decoded_tokens.items():
            t_clean = text.strip()

            if t_clean and all(c in "0123456789.-"
                               for c in t_clean) and ".." not in t_clean:
                self.tokens_numbers.append(tid)

            # Filter for Strings (removes line breaks that corrupt JSON)
            if not any(ch in text for ch in ["\n", "\r", "Ċ", "$"]):
                if text.strip() != '"':
                    self.tokens_strings.append(tid)

    def _setup_control_tokens(self) -> None:
        """Encodes structural JSON tokens ({, }, \", spaces, booleans)."""
        self.t_quote = self.llm.encode('"').tolist()[0][0]
        self.t_cb = self.llm.encode("}").tolist()[0][0]
        self.t_sp = self.llm.encode(" ").tolist()[0][0]
        self.final_end = self.llm.encode('\n  }\n}').tolist()[0]

        self.param_key_tokens = {}
        for func in self.functions:
            fname = func["name"]
            self.param_key_tokens[fname] = []
            for i, pname in enumerate(list(func.get("parameters", {}).keys())):
                sep = f'\n    "{pname}": ' if i == 0 else f',\n    "{pname}": '
                self.param_key_tokens[fname].append(
                    self.llm.encode(sep).tolist()[0]
                )

        self.t_true_seq = self.llm.encode("true").tolist()[0]
        self.t_false_seq = self.llm.encode("false").tolist()[0]

    def reset_prompt(self, prompt_text: str) -> None:
        """Resets the FSM state for a new user prompt."""
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

        prompt_escaped = prompt_text.replace('\\', '\\\\').replace('"', '\\"')
        self.function_prefixes = {}
        for func in self.functions:
            p = (
                f'{{\n  "prompt": "{prompt_escaped}",\n'
                f'  "name": "{func["name"]}",\n  "parameters": {{'
            )
            self.function_prefixes[func["name"]] = (
                self.llm.encode(p).tolist()[0]
            )

    def get_allowed_tokens(
        self, generated_ids: List[int], prompt_text: str
    ) -> List[int]:
        """FSM Engine: Decides the next allowed tokens."""
        if self.state == "COMPLETED":
            return [self.t_sp]

        # UPDATE FSM STATE
        if generated_ids:
            last = generated_ids[-1]
            if self.state == "PREFIX":
                for fname, pfix in self.function_prefixes.items():
                    if (
                        generated_ids[-len(pfix):] == pfix
                        or generated_ids == pfix
                    ):
                        self.current_func = self.function_map[fname]
                        self.param_names = list(self.current_func.get(
                            "parameters", {}).keys())
                        self.current_param_idx = 0
                        if self.param_names:
                            self.state = "NEXT_PARAM"
                        else:
                            self.state = "END"
                        break
            elif self.state == "NEXT_PARAM":
                if self.current_func is None:
                    return [self.t_cb]
                keys = self.param_key_tokens[self.current_func["name"]][
                    self.current_param_idx]
                if (
                    self.key_progress < len(keys)
                    and last == keys[self.key_progress]
                ):
                    self.key_progress += 1
                    if self.key_progress == len(keys):
                        self.state = "VALUE"
                        self.string_sub_state = 0
                        self.current_string_val = ""
                        self.current_len = 0
            elif self.state == "VALUE":
                if self.current_func is None:
                    return [self.t_cb]
                param_type = self.current_func["parameters"][self.param_names[
                    self.current_param_idx]].get("type", "string")
                if param_type not in ["number", "integer", "boolean"]:
                    param_type = "string"

                if param_type == "string":
                    if self.string_sub_state == 0 and last == self.t_quote:
                        self.string_sub_state = 1
                    elif self.string_sub_state == 1:
                        if last == self.t_quote:
                            self.current_param_idx += 1
                            if self.current_param_idx < len(self.param_names):
                                self.state = "NEXT_PARAM"
                            else:
                                self.state = "END"
                            self.key_progress = 0
                        else:
                            self.current_string_val += self.decoded_tokens.get(
                                last, "")
                            self.current_len += 1
                elif param_type in ["number", "integer"]:
                    is_last = self.current_param_idx == len(
                        self.param_names) - 1
                    if is_last:
                        target_end = self.final_end[0]
                    else:
                        func_name = self.current_func["name"]
                        target_end = self.param_key_tokens[func_name][
                            self.current_param_idx + 1][0]
                    if last == target_end:
                        if is_last:
                            self.state = "END"
                            self.end_progress = 1
                        else:
                            self.current_param_idx += 1
                            self.state = "NEXT_PARAM"
                            self.key_progress = 1
                    else:
                        self.current_len += 1
                elif param_type == "boolean":
                    if self.bool_chosen is None:
                        if last == self.t_true_seq[0]:
                            self.bool_chosen = "true"
                            self.bool_progress = 1
                        elif last == self.t_false_seq[0]:
                            self.bool_chosen = "false"
                            self.bool_progress = 1
                    else:
                        if not self.bool_finished:
                            if self.bool_chosen == "true":
                                target = self.t_true_seq
                            else:
                                target = self.t_false_seq
                            if (
                                self.bool_progress < len(target)
                                and last == target[self.bool_progress]
                            ):
                                self.bool_progress += 1
                                if self.bool_progress == len(target):
                                    self.bool_finished = True
                    if self.bool_finished:
                        is_last = self.current_param_idx == len(
                            self.param_names) - 1
                        if is_last:
                            target_end = self.final_end[0]
                        else:
                            func_name = self.current_func["name"]
                            target_end = self.param_key_tokens[func_name][
                                self.current_param_idx + 1][0]
                        if last == target_end:
                            if is_last:
                                self.state = "END"
                                self.end_progress = 1
                            else:
                                self.current_param_idx += 1
                                self.state = "NEXT_PARAM"
                                self.key_progress = 1
            elif self.state == "END":
                if (
                    self.end_progress < len(self.final_end)
                    and last == self.final_end[self.end_progress]
                ):
                    self.end_progress += 1
                    if self.end_progress == len(self.final_end):
                        self.state = "COMPLETED"

        # TOKEN GENERATION (Delegated to private helpers)
        if self.state == "COMPLETED":
            return [self.t_sp]

        if self.state == "PREFIX":
            step = len(generated_ids)
            allowed = {
                p[step]
                for p in self.function_prefixes.values()
                if step < len(p) and p[:step] == generated_ids
            }
            return list(allowed) if allowed else [self.t_cb]

        if self.state == "NEXT_PARAM":
            if self.current_func is None:
                return [self.t_cb]
            return [self.param_key_tokens[self.current_func["name"]][
                self.current_param_idx][self.key_progress]]

        if self.state == "VALUE":
            return self._get_value_tokens(generated_ids)

        if self.state == "END":
            if self.end_progress < len(self.final_end):
                return [self.final_end[self.end_progress]]
            return [self.t_sp]

        return [self.t_cb]

    # ==========================================
    # GENERATION MODULES (Private Helpers)
    # ==========================================

    def _get_value_tokens(self, generated_ids: List[int]) -> List[int]:
        """Routes token generation logic based on the variable type."""
        if self.current_func is None:
            return [self.t_cb]
        pname = self.param_names[self.current_param_idx]
        param_type = self.current_func["parameters"][pname].get("type",
                                                                "string")
        if param_type not in ["number", "integer", "boolean"]:
            param_type = "string"

        is_last = self.current_param_idx == len(self.param_names) - 1
        target_end = self.final_end[0] if is_last else self.param_key_tokens[
            self.current_func["name"]][self.current_param_idx + 1][0]

        if param_type == "string":
            if self.string_sub_state == 0:
                return [self.t_quote]

            is_regex = "regex" in pname.lower() or "pattern" in pname.lower()
            if is_regex:
                return self._generate_regex_string(generated_ids)
            else:
                return self._generate_extraction_string()

        elif param_type in ["number", "integer"]:
            if self.current_len >= 12:
                return [target_end]
            allowed = list(self.tokens_numbers)
            if self.current_len > 0:
                allowed.append(target_end)
            return allowed

        elif param_type == "boolean":
            if self.bool_chosen is None:
                return [self.t_true_seq[0], self.t_false_seq[0]]

            if not self.bool_finished:
                if self.bool_chosen == "true":
                    target = self.t_true_seq
                else:
                    target = self.t_false_seq

                return [target[self.bool_progress]]

            return [target_end]

        # fallback (shouldn't happen) -- return closing brace
        return [self.t_cb]

    def _generate_regex_string(self, generated_ids: List[int]) -> List[int]:
        """Generates a strict whitelist exclusively for regular expressions."""
        if self.current_len >= 7:
            return [self.t_quote]

        allowed_chars = set("[]+-0123456789abcdefghijklmnopq"
                            "rstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        last_token = self.decoded_tokens.get(
            generated_ids[-1], "").strip() if generated_ids else ""
        allowed = [self.t_quote]

        for tid in self.tokens_strings:
            tok_str = self.decoded_tokens[tid]

            if '"' in tok_str and tok_str != '\\"':
                continue
            if not all(c in allowed_chars for c in tok_str.strip()):
                continue
            if tok_str.strip() == last_token and len(tok_str.strip()) > 0:
                continue

            allowed.append(tid)
        return allowed

    def _generate_extraction_string(self) -> List[int]:
        """Forces the model to extract exact values copied from the prompt."""
        allowed = [self.t_quote]
        if self.current_len >= 35:
            return allowed

        for tid in self.tokens_strings:
            tok_str = self.decoded_tokens[tid]
            if "\n" in tok_str or "\r" in tok_str or "Ċ" in tok_str:
                continue
            if '"' in tok_str and tok_str != '\\"':
                continue

            candidate = self.current_string_val + tok_str
            candidate_clean = candidate.replace('\\"', '"').replace('\\\\',
                                                                    '\\')

            if candidate_clean in self.prompt_text:
                allowed.append(tid)

        return allowed
