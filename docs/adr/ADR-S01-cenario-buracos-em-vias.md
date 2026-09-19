# ADR-S01 — Cenário: Cidades Inteligentes → detecção e segmentação de buracos em vias

**Status:** Aceito
**Base:** enunciado da Sistematização (cenário "Cidades Inteligentes: buracos em vias, vagas de estacionamento ou análise de tráfego urbano"); doutrina em ADR-008, ADR-009, ADR-019

## Contexto

O cenário Cidades Inteligentes admite três sub-problemas. O prazo é de dois dias e o barema exige, no mesmo domínio, detecção fine-tuned (25%), segmentação (20%) e vídeo real (10%). O sub-problema precisa ter dataset público com anotações que sirvam às duas tarefas, e um vídeo fácil de obter.

| Sub-problema | Dataset público com polígonos? | Vídeo real fácil? | Segmentação faz sentido? |
|---|---|---|---|
| **Buracos em vias** | Sim — vários no Roboflow Universe, CC BY 4.0, 300–5 000 imagens | Sim — celular no carro ou dashcam pública | Sim — a máscara mede a **área/extensão** do dano, que a caixa não dá |
| Vagas de estacionamento | Poucos com polígonos; maioria só caixas (PKLot) | Requer câmera fixa elevada | Fraca — vaga é retângulo; máscara pouco acrescenta |
| Tráfego urbano | Caixas abundantes (COCO já cobre carro/pessoa) | Sim | Fraca no prazo — fine-tuning sobre COCO acrescenta pouco e a segmentação de instância de veículos é pesada |

## Decisão

**Buracos em vias**, classe única `pothole`. Problema de negócio: priorização de manutenção viária — um veículo instrumentado (ou o cidadão, pelo celular) percorre as ruas; o sistema detecta buracos e, pela máscara, estima a área relativa de cada um, permitindo ranquear ordens de serviço.

## Alternativas rejeitadas

- **Vagas de estacionamento:** a segmentação seria artificial; datasets com máscaras são raros.
- **Tráfego:** o valor do fine-tuning é baixo (COCO já detecta veículos) e a análise crítica ficaria pobre.

## Consequências

- Um único dataset com polígonos alimenta as duas tarefas (ADR-S02, S04).
- A comparação "caixas × máscaras" tem argumento natural: a caixa superestima a área de buracos alongados/diagonais; a máscara segue o contorno.
- Modos de falha previsíveis e ricos para a análise de erros: sombras, poças, remendos de asfalto, tampas de bueiro, rachaduras, buracos pequenos ao longe, borda da imagem.
- Ética (ADR-019): não há pessoas como alvo; se o vídeo próprio captar placas ou rostos, borrar antes de publicar.

## Relacionados

ADR-S02, ADR-S04, SPEC-S01.
