from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

def texto_para_expressao(texto):
    # Ajustando o prompt com mais exemplos de problemas matemáticos
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
        "Example 6:\n"
        f"Sentence: {texto}\n"
        "Expression:"
    )
    result = generator(few_shot_prompt, max_new_tokens=30, do_sample=False)
    return result[0]["generated_text"]

# Interação
print("🔢 Conversor de linguagem natural para expressão matemática (em inglês). Digite 'sair' para encerrar.")
while True:
    entrada = input("Frase matemática: ")
    if entrada.lower() in ['sair', 'exit', 'quit']:
        print("Encerrando.")
        break
    expressao = texto_para_expressao(entrada)
    print(f"🧠 Expressão gerada: {expressao}\n")
