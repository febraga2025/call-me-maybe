# Relatório de problemas na saída JSON

## Contexto observado

A execução gerou JSONs que passaram na etapa de gravação, mas falharam na moulinette em vários testes. O problema principal não é só o formato JSON: a IA está extraindo os argumentos de forma incorreta em alguns casos e adicionando conteúdo que não existe no prompt.

## Problemas encontrados

### 1) O JSON parece válido, mas os valores estão errados

O caso mais claro é o teste 7. A chamada esperada era:

```text
fn_execute_sql_query({'query': 'INSERT INTO logs VALUES (1, 2, 3)', 'database': 'system'})
```

Mas a IA produziu:

```text
fn_execute_sql_query({'query': 'INSERT INTO logs VALUES (1, 2, 3) ON THE SYSTEM DATABASE', 'database': 'system database'})
```

Isso mostra duas falhas:

1. O campo `query` foi contaminado com texto extra do prompt.
2. O campo `database` foi expandido além do valor esperado.

### 2) Caminhos de arquivo perderam caracteres importantes

No teste 8, o esperado era:

```text
'/home/user/data.json'
```

Mas a IA gerou:

```text
'home/user/data.json'
```

Ou seja, o caractere `/` inicial foi removido. Isso quebra a semântica do caminho e passa uma string formalmente válida, mas funcionalmente errada.

### 3) Strings longas e repetitivas estão escapando do controle

Nos testes 10 e 11, a IA acrescentou texto extra dentro do template:

```text
Hello {user}'s profile! 1234567890123456789012345678901234567
```

e

```text
Say "hello" to {name} and {age} years old. Please say hello to {name} again...
```

Isso indica que o gerador está permitindo continuação demais dentro de strings, sem uma regra forte para parar na fronteira correta do valor extraído.

### 4) O sistema salva o JSON mesmo quando a IA está semanticamente errada

O programa imprime `Sucesso!` no final, mas a moulinette mostra que apenas 7 de 11 testes passaram. Isso significa que a validação local está aceitando a estrutura, mas não está garantindo fidelidade semântica ao prompt.

### 5) O problema não é só formato, é extração

Os JSONs não estão quebrando por sintaxe. Eles quebram porque os campos `query`, `path` e `template` estão sendo preenchidos com conteúdo que não deveria entrar.

Isso é típico de um modelo que:

1. entende parcialmente o prompt,
2. continua a gerar texto além do necessário,
3. não é barrado por uma checagem semântica forte no final.

## Sugestões de melhoria

### Prioridade alta

1. Validar a estrutura final do JSON, mas também o conteúdo de cada campo.
2. Rejeitar strings que contenham texto extra que não pertence ao trecho extraído do prompt.
3. Abortar a geração quando o modelo ultrapassar um limite lógico de conteúdo para o parâmetro.
4. Fazer a validação final comparar o valor gerado com o trecho esperado do prompt, quando possível.

### Prioridade média

1. Melhorar o motor para parar a string exatamente no ponto certo, sem depender só de limite de tamanho.
2. Para campos de caminho, preservar prefixos críticos como `/` e barras invertidas `\\`.
3. Para campos livres como `template`, evitar que o modelo complete frases ou acrescente explicações.
4. Separar melhor parâmetros numéricos, caminhos e texto livre dentro das regras do gerador.

### Prioridade baixa, mas útil

1. Adicionar testes negativos com strings curtas e caminhos absolutos.
2. Registrar quando o modelo adiciona tokens que não pertencem ao valor extraído.
3. Medir quantas vezes o fallback do engine é acionado durante a geração.

## Diagnóstico resumido

O sistema está passando JSONs formalmente válidos, mas semanticamente errados. A moulinette reprova porque o conteúdo dos parâmetros não bate com o que o teste espera.

O ajuste mais importante é reforçar a contenção das strings e impedir que o modelo complemente valores além do trecho realmente presente no prompt.