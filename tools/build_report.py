"""Monta o relatório: Markdown → HTML → PDF, com o apêndice de código gerado do repositório.

    python tools/build_report.py                 # PDF do relatório (6–10 páginas, limite do enunciado)
    python tools/build_report.py --com-apendice  # PDF completo, com o apêndice de código

O apêndice é sempre regenerado dentro de reports/relatorio.md (versão do repositório); no PDF de
entrega ele fica de fora para respeitar o limite de páginas.

O PDF é o entregável. A folha de estilo, a ordem das seções e o apêndice são derivados do
repositório; nada é montado à mão. O apêndice lista o código que produziu os resultados —
gerado a partir dos arquivos, não transcrito, para que o relatório nunca descreva um código
que já não existe.

A conversão final usa o Google Chrome em modo headless (única ferramenta de impressão presente).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import markdown

RAIZ = Path(__file__).resolve().parent.parent
RELATORIO = RAIZ / "reports" / "relatorio.md"
HTML_SAIDA = RAIZ / "reports" / "relatorio.html"
PDF_SAIDA = RAIZ / "reports" / "relatorio.pdf"
TITULO = "Detecção e segmentação de buracos em vias urbanas com YOLO11"

MARCA_INICIO = "<!-- INICIO-APENDICE-CODIGO -->"

# Ordem do pipeline, não alfabética: do dado bruto ao relatório.
ARQUIVOS_APENDICE = [
    ("A. Dados", ["data.yaml", "scripts/baixar_dataset.py", "scripts/filtrar_classes.py", "scripts/eda.py"]),
    ("B. Refino das máscaras com SAM", ["scripts/refinar_mascaras_sam.py"]),
    ("C. Treino", ["scripts/treinar.py", "scripts/pipeline.sh"]),
    ("D. Avaliação", ["scripts/avaliar.py"]),
    ("E. Relatório e ambiente", ["tools/build_report.py", "scripts/gerar_notebook.py", "requirements.txt", "requirements-report.txt"]),
]
LINGUAGEM = {".py": "python", ".sh": "bash", ".yaml": "yaml", ".yml": "yaml", ".txt": "text", ".md": "markdown"}

ESTILO = """
@page { size: A4; margin: 14mm 15mm 15mm; }
* { box-sizing: border-box; }
body { font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; font-size: 9.5pt; line-height: 1.38;
       color: #16202B; background: #FFF; margin: 0; padding: 0; }
h1 { font-size: 22pt; line-height: 1.2; margin: 0 0 4pt; letter-spacing: -.01em; }
h2 { font-size: 12.5pt; margin: 11pt 0 5pt; padding-bottom: 4pt; border-bottom: .8pt solid #C9D2DB; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 10pt 0 4pt; page-break-after: avoid; }
h4 { font-size: 10pt; margin: 12pt 0 4pt; font-family: "IBM Plex Mono", monospace; color: #47576B; page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
p { margin: 0 0 5pt; }
ul, ol { margin: 0 0 8pt; padding-left: 18pt; }
li { margin-bottom: 2.5pt; }
strong { font-weight: 600; }
hr { border: none; border-top: .8pt solid #C9D2DB; margin: 9pt 0; }
blockquote { margin: 9pt 0; padding: 7pt 12pt; border-left: 2.5pt solid #4A6E8A; background: #F2F5F8; font-size: 9.8pt; }
blockquote p:last-child { margin-bottom: 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 9pt; font-size: 8.8pt; page-break-inside: auto; }
th, td { border: .6pt solid #C9D2DB; padding: 4pt 7pt; text-align: left; vertical-align: top; }
th { background: #EEF2F6; font-weight: 600; }
td:nth-child(n+2) { font-variant-numeric: tabular-nums; }
code { font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace; font-size: .87em; background: #F0F3F6; padding: .5pt 3pt; border-radius: 2pt; }
pre { background: #F7F9FB; border: .6pt solid #D8E0E8; border-radius: 3pt; padding: 5pt 8pt; overflow-x: auto;
      font-size: 7.4pt; line-height: 1.35; margin: 5pt 0 8pt; page-break-inside: auto; }
pre code { background: none; padding: 0; font-size: inherit; }
img { max-width: 70%; height: auto; display: block; margin: 5pt auto; page-break-inside: avoid; }
p img[alt="CEUB"] { max-width: 30%; }
p img[alt^="Matriz de confusão"] { max-width: 40%; }
p img[alt^="Precisão × revocação"], p img[alt^="Razão entre"] { max-width: 48%; }
p img[alt^="Controle de qualidade"], p img[alt^="Caixas do detector"], p img[alt^="Falsos"] { max-width: 84%; }
h1 + h3 + table { margin-bottom: 12pt; }
h2#apêndice--código-fonte { page-break-before: always; }
"""


def commit_curto() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "sem-git"


def apendice() -> str:
    """Gera o apêndice a partir dos arquivos do repositório (nunca transcrito)."""
    partes, n_arq, n_lin = [], 0, 0
    for secao, arquivos in ARQUIVOS_APENDICE:
        partes.append(f"\n### {secao}\n")
        for rel in arquivos:
            p = RAIZ / rel
            if not p.exists():
                continue
            texto = p.read_text(encoding="utf-8", errors="replace").rstrip("\n")
            linhas = texto.count("\n") + 1
            n_arq += 1; n_lin += linhas
            partes.append(f"\n#### `{rel}` · {linhas} linhas\n```{LINGUAGEM.get(p.suffix, 'text')}\n{texto}\n```\n")
    cabeca = (f"\n## Apêndice — Código-fonte\nListagem integral do código que produziu os resultados deste relatório, no commit "
              f"`{commit_curto()}`. As seções seguem a ordem do pipeline — do dado bruto ao relatório — e não a ordem alfabética.\n\n"
              "Este apêndice é **gerado a partir dos arquivos do repositório**, não transcrito: código copiado para dentro de um documento "
              "diverge do original no primeiro ajuste.\n\n"
              f"**{n_arq} arquivos · {n_lin:,} linhas.**\n".replace(",", "."))
    return cabeca + "".join(partes)


def aplicar_apendice() -> None:
    texto = RELATORIO.read_text(encoding="utf-8")
    if MARCA_INICIO not in texto:
        texto = texto.rstrip("\n") + f"\n\n---\n\n{MARCA_INICIO}\n"
    corpo = texto.split(MARCA_INICIO)[0]
    RELATORIO.write_text(corpo + MARCA_INICIO + "\n" + apendice(), encoding="utf-8")


def construir_html(com_apendice: bool) -> str:
    texto = RELATORIO.read_text(encoding="utf-8")
    if not com_apendice:
        texto = texto.split(MARCA_INICIO)[0].rstrip("\n").rstrip("-").rstrip("\n") + "\n"
    corpo = markdown.markdown(texto, extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list"],
                              extension_configs={"codehilite": {"guess_lang": False, "noclasses": True, "pygments_style": "friendly"}})
    return ('<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
            f"<title>{TITULO}</title>"
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap">'
            f"<style>{ESTILO}</style></head><body>{corpo}</body></html>")


def chrome() -> str:
    for nome in ("google-chrome", "chromium", "chromium-browser", "google-chrome-stable"):
        caminho = shutil.which(nome)
        if caminho:
            return caminho
    raise SystemExit("Nenhum navegador encontrado para gerar o PDF.")


def main() -> int:
    com_apendice = "--com-apendice" in sys.argv
    aplicar_apendice()
    HTML_SAIDA.write_text(construir_html(com_apendice), encoding="utf-8")
    print(f"HTML  → {HTML_SAIDA.relative_to(RAIZ)}  ({HTML_SAIDA.stat().st_size / 1024:.0f} KB)")
    resultado = subprocess.run(
        [chrome(), "--headless", "--disable-gpu", "--no-sandbox", "--run-all-compositor-stages-before-draw",
         "--virtual-time-budget=20000", "--no-pdf-header-footer", f"--print-to-pdf={PDF_SAIDA}", HTML_SAIDA.as_uri()],
        capture_output=True, text=True, timeout=600)
    if not PDF_SAIDA.exists():
        print(resultado.stderr[-1500:], file=sys.stderr)
        raise SystemExit("Falha ao gerar o PDF.")
    print(f"PDF   → {PDF_SAIDA.relative_to(RAIZ)}  ({PDF_SAIDA.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
