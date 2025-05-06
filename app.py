from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# === MODELO DE TRADUÇÃO (PT → EN) COM NLLB ===
nllb_model_name = "facebook/nllb-200-distilled-600M"
nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

# Função de tradução com NLLB corrigida
def traduzir_para_ingles(texto_pt):
    nllb_tokenizer.src_lang = "por_Latn"  # Define idioma de entrada
    inputs = nllb_tokenizer(texto_pt, return_tensors="pt", padding=True)

    # ID do token de início para inglês
    eng_token_id = nllb_tokenizer.convert_tokens_to_ids(">>eng_Latn<<")

    translated_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=eng_token_id
    )

    texto_en = nllb_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return texto_en

# === MODELO DE GERAÇÃO DE EXPRESSÃO ===
flan_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def texto_para_expressao(texto):
    few_shot_prompt = (
        "Convert the sentence into a math expression using only numbers and operators.\n"
        "Example 1:\n"
        "Sentence: The sum of twice 5 and 4\n"
        "Expression: 2 * 5 + 4\n"
        "Example 2:\n"
        "Sentence: Three times the number 6 minus 2\n"
        "Expression: 3 * 6 - 2\n"
        "Example 3:\n"
        "Sentence: Twice the number 4 plus 6\n"
        "Expression: 2 * 4 + 6\n"
        "Example 4:\n"
        "Sentence: Double 7 minus 3\n"
        "Expression: 2 * 7 - 3\n"
        "Example 5:\n"
        "Sentence: The product of 5 and the difference between 9 and 4\n"
        "Expression: 5 * (9 - 4)\n"
        f"Sentence: {texto}\n"
        "Expression:"
    )
    result = generator(few_shot_prompt, max_new_tokens=30, do_sample=False)
    return result[0]["generated_text"]

# === AVALIAÇÃO DA EXPRESSÃO ===
def resolver_expressao(expr):
    try:
        resultado = eval(expr)
        return resultado
    except Exception as e:
        return f"Erro ao resolver: {e}"

# === INTERAÇÃO COM O USUÁRIO ===
print("🔢 Conversor de linguagem natural para expressão matemática (em português ou inglês). Digite 'sair' para encerrar.")
while True:
    entrada = input("Frase matemática: ")
    if entrada.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando.")
        break

    frase_em_ingles = traduzir_para_ingles(entrada)
    expressao = texto_para_expressao(frase_em_ingles)
    resultado = resolver_expressao(expressao)

    print(f"🧠 Expressão gerada: {expressao}")
    print(f"📊 Resultado: {resultado}\n")
