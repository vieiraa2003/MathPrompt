# 🧮 Conversor de Linguagem Natural para Expressões Matemáticas

Este projeto usa inteligência artificial (Transformers + Flask) para converter frases simples (em português ou inglês) em expressões matemáticas e resolver o resultado.

## 🚀 Tecnologias

- Python 3.10+
- Flask (para interface web)
- Transformers (modelo Flan-T5)
- Torch (PyTorch)
- NLLB (tradução automática)

## 🛠️ Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seuusuario/seuprojeto.git
cd seuprojeto

2. Crie e ative um ambiente virtual (opcional, mas recomendado):

bash
Copiar
Editar

3. Instale as dependências:

python -m pip install -r requirements.txt

4. Rode o aplicativo:

python app.py

Acesse no navegador: http://127.0.0.1:5000

✨ Exemplos de entrada
o dobro de 8

a soma de 2 mais 2

a metade de 6

2 mais 3

4 vezes 5

a diferença entre 7 e 3

🔒 Limitações
Suporta apenas soma, subtração e multiplicação (divisão desativada por ora).

Apenas frases simples em linguagem natural.

Projeto em desenvolvimento incremental.

📁 Estrutura

/static
  style.css
/templates
  index.html          → interface HTML do app
app.py                → app Flask
conversor.py          → lógica de IA + tradução
requirements.txt      → dependências
README.md             → este arquivo

📜 Licença
MIT - sinta-se livre para modificar, estudar e usar!