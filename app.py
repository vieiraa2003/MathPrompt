from flask import Flask, render_template, request
from conversor import traduzir_para_ingles, texto_para_expressao, resolver_expressao

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = ""
    expressao = ""
    entrada = ""

    if request.method == "POST":
        entrada = request.form["frase"]
        frase_en = traduzir_para_ingles(entrada)
        expressao = texto_para_expressao(frase_en)
        resultado = resolver_expressao(expressao)

    return render_template("index.html", entrada=entrada, expressao=expressao, resultado=resultado)

if __name__ == "__main__":
    app.run(debug=True)
