from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import re

# === MODELO DE TRADUÇÃO (PT → EN) COM NLLB ===
nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

def traduzir_para_ingles(texto_pt):
    nllb_tokenizer.src_lang = "por_Latn"
    inputs = nllb_tokenizer(texto_pt, return_tensors="pt", padding=True)
    eng_token_id = nllb_tokenizer.convert_tokens_to_ids(">>eng_Latn<<")
    translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=eng_token_id)
    return nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# === MODELO DE GERAÇÃO DE EXPRESSÃO ===
flan_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def texto_para_expressao(texto):
    few_shot_prompt = (
        "Convert the sentence into a math expression using only numbers and the +, -, * operators.\n"
        "# Exemplo 1: 2 mais 2\n"
        "Sentence: 2 plus 2\n"
        "Expression: 2 + 2\n"
        "# Exemplo 2: 4 menos 3\n"
        "Sentence: 4 minus 3\n"
        "Expression: 4 - 3\n"
        "# Exemplo 3: 5 vezes 6\n"
        "Sentence: 5 times 6\n"
        "Expression: 5 * 6\n"
        "# Exemplo 4: O dobro de 7\n"
        "Sentence: Double of 7\n"
        "Expression: 2 * 7\n"
        "# Exemplo 5: A soma de 3 mais 4\n"
        "Sentence: The sum of 3 and 4\n"
        "Expression: 3 + 4\n"
        "# Exemplo 6: A diferença entre 9 e 5\n"
        "Sentence: The difference between 9 and 5\n"
        "Expression: 9 - 5\n"
        f"Sentence: {texto}\n"
        "Expression:"
    )
    result = generator(few_shot_prompt, max_new_tokens=30, do_sample=False)
    return result[0]["generated_text"]

# === AVALIAÇÃO SEGURA ===
def resolver_expressao(expr):
    try:
        if not re.fullmatch(r"[0-9\+\-\*\(\) ]+", expr):
            raise ValueError("Expressão inválida")
        return eval(expr)
    except Exception as e:
        return f"Erro ao resolver: {e}"

# === NORMALIZAÇÃO ===
def normalizar(expr):
    return re.sub(r'\s+', '', expr.strip())

# === TESTE DE ACURÁCIA ===
def testar_acuracia():
    testes = [
        ("2 mais 2", "2 + 2"),
        ("4 menos 3", "4 - 3"),
        ("5 vezes 6", "5 * 6"),
        ("O dobro de 7", "2 * 7"),
        ("A soma de 3 mais 4", "3 + 4"),
        ("A diferença entre 9 e 5", "9 - 5"),
    ]

    acertos = 0
    for pt, esperado in testes:
        en = traduzir_para_ingles(pt)
        gerado = texto_para_expressao(en).strip()
        if normalizar(gerado) == normalizar(esperado):
            acertos += 1
        else:
            print(f"❌ Frase: {pt}")
            print(f"   Esperado: {esperado}")
            print(f"   Gerado:   {gerado}\n")

    precisao = acertos / len(testes)
    print(f"\n🎯 Precisão do modelo com {len(testes)} testes: {precisao:.2%}\n")

# === INTERAÇÃO ===
if __name__ == "__main__":
    print("🔢 Conversor de linguagem natural para expressão matemática (sem divisão).")
    print("Digite 'avaliar' para medir precisão ou 'sair' para encerrar.\n")

    while True:
        entrada = input("Frase matemática: ")
        if entrada.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando.")
            break
        elif entrada.lower() == "avaliar":
            testar_acuracia()
            continue

        frase_em_ingles = traduzir_para_ingles(entrada)
        expressao = texto_para_expressao(frase_em_ingles)
        resultado = resolver_expressao(expressao)

        print(f"🧠 Expressão gerada: {expressao}")
        print(f"📊 Resultado: {resultado}\n")
