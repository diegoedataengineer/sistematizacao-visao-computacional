# ADR-S07 — Ambiente: Colab (T4) para treino, Google Drive como armazenamento persistente (sincronizado por rclone), GitHub pessoal como repositório de entrega

**Status:** Aceito
**Base:** ferramentas sugeridas no enunciado (Colab GPU, PyTorch, Ultralytics, OpenCV, GitHub); configuração já feita nesta máquina (rclone + `sync-colab.sh`)

## Decisão

| Componente | Escolha | Detalhe |
|---|---|---|
| Treino/inferência | **Google Colab, GPU T4** | `Ambiente de execução → Alterar tipo → T4 GPU`; conferir `!nvidia-smi` |
| Persistência | **Google Drive** montado em `/content/drive/MyDrive/visao-computacional/sistematizacao/` | `project=` do Ultralytics aponta para o Drive → `runs/` sobrevive à queda da sessão; `resume=True` retoma |
| Sincronia local | `rclone bisync` já configurado (`./sync-colab.sh`) | traz `runs/`, figuras e notebook para esta máquina; **excluir pesos > 100 MB do Git** |
| Código e entrega | **GitHub, conta pessoal** (`git@github-pessoal:diegoedataengineer/vcrp-buracos-vias.git`) | `git config user.name "Diego Nunes de Morais"`, `user.email diego.dataengineer1987@gmail.com` no repo; README com nomes de todos os integrantes |
| Pesos | Drive (link público de leitura no README) ou GitHub Releases | `best.pt` de det e seg (~20 MB cada em `s`) cabem em Release |
| Dataset | link do Roboflow + `data.yaml` versionado; imagens **não** no Git | reprodução por download com a chave de API do usuário ou pelo zip público |
| Fallback local | GTX 1060 6 GB + `torch` CUDA | treina `yolo11n/s` em 640 com `batch=8`; ~3–4× mais lento que a T4 |

Estrutura do Drive/repo em SPEC-S06.

## Consequências

- Uma queda de sessão do Colab custa minutos, não horas.
- O notebook final roda em qualquer conta Colab a partir do `data.yaml` e do zip do dataset — critério de reprodutibilidade (10%).

## Relacionados

SPEC-S02, SPEC-S06; `sync-colab.sh` na raiz do projeto.
