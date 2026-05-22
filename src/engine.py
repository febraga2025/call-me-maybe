import json

class ConstrainedEngine:
    def __init__(self, llm, functions):
        self.llm = llm
        self.functions = functions

        # Carregar vocabulário
        vocab_path = self.llm.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        self.all_token_ids = list(self.vocab.values())
        
        # Caixinhas de tokens filtrados
        self.tokens_numeros = set()
        self.tokens_strings = set()

        # Preparar os filtros no início
        for texto, token_id in self.vocab.items():
            texto_limpo = texto.replace("Ġ", "")
            
            # Filtro para números
            if texto_limpo.isdigit() or texto_limpo in [".", "-", ""]:
                self.tokens_numeros.add(token_id)
                
            # Filtro para strings (tudo que não seja " ou })
            if '"' not in texto and '}' not in texto:
                self.tokens_strings.add(token_id)

        # Adicionar tokens de controle essenciais para fechar o JSON
        self.tokens_numeros.update([
            self.llm.encode(',').tolist()[0][0],
            self.llm.encode('\n').tolist()[0][0],
            self.llm.encode('}').tolist()[0][0],
            self.llm.encode(' ').tolist()[0][0]
        ])

    def get_allowed_tokens(self, gerados_ids: list[int], pergunta_atual: str) -> list[int]:
        texto_gerado = self.llm.decode(gerados_ids)
        
        # 1. FASE DE ESCOLHA DA FUNÇÃO
        if '"parameters": {' not in texto_gerado:
            caminhos_ids = []
            for func in self.functions:
                texto_caminho = f'{{\n  "prompt": "{pergunta_atual}",\n  "name": "{func["name"]}",\n  "parameters": {{'
                ids = self.llm.encode(texto_caminho).tolist()[0]
                caminhos_ids.append(ids)

            passo_atual = len(gerados_ids)
            tokens_permitidos = set()
            
            for caminho in caminhos_ids:
                if passo_atual < len(caminho) and caminho[:passo_atual] == gerados_ids:
                    tokens_permitidos.add(caminho[passo_atual])
            
            if tokens_permitidos:
                return list(tokens_permitidos)

        # 2. FASE DOS PARÂMETROS
        funcao_escolhida = None
        for func in self.functions:
            if f'"name": "{func["name"]}"' in texto_gerado:
                funcao_escolhida = func
                break
                
        if funcao_escolhida:
            # Verifica se estamos a preencher o VALOR de um parâmetro (depois do ": ")
            if ': ' in texto_gerado.splitlines()[-1]:
                partes = texto_gerado.split('"')
                if len(partes) >= 2:
                    param_nome = partes[-2]
                    regra = funcao_escolhida["parameters"].get(param_nome, {})
                    tipo = regra.get("type", "")
                    
                    if tipo == "number":
                        # Cria uma cópia da caixinha de números
                        tokens_permitidos = self.tokens_numeros.copy()
                        
                        # Lógica para evitar a "alucinação de zeros" ou vírgulas antecipadas
                        # Se já existe um dígito, permitimos vírgula e fecho de chaveta
                        if any(char.isdigit() for char in texto_gerado.split()[-1]):
                            # O modelo já tem a liberdade de fechar o parâmetro
                            pass 
                        return list(tokens_permitidos)
                        
                    elif tipo == "string":
                        return list(self.tokens_strings)
        
        # Fallback para o resto do JSON (fecho de chaves, etc)
        return self.all_token_ids