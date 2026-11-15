# Detector de pneumonia em raio-X 🫁

Classificador de imagens que recebe um raio-X de tórax e responde **normal** ou **pneumonia**, com um app em Streamlit para testar por upload ou pela câmera. Projeto educacional: **não é diagnóstico médico**.

## O problema

Ferramentas sem código, como o Teachable Machine, treinam um classificador de imagens em minutos, mas escondem o que mais importa numa aplicação de saúde: como o modelo foi avaliado e onde ele erra. Este projeto refaz a ideia em Python para mostrar o caminho inteiro, do banco de imagens ao app, com as métricas à vista. Serve para estudantes e professores que queiram entender e demonstrar como funciona a classificação de imagens com *transfer learning*.

## Como funciona

1. **Dados.** 2.772 raio-X do banco público Chest X-Ray Images (Pneumonia): 1.341 normais e 1.431 com pneumonia.
2. **Divisão por paciente.** As imagens são separadas em treino (1.945), validação (436) e teste (391) de modo que a mesma pessoa nunca apareça em dois conjuntos. Sem esse cuidado, o modelo seria testado em pacientes que já viu, e a acurácia sairia inflada.
3. **Modelo.** MobileNetV2 pré-treinada na ImageNet (*transfer learning*), com entrada de 224 x 224 pixels.
4. **Limiar de decisão.** O ponto de corte entre normal e pneumonia é escolhido na validação e salvo em `limiar.json` (ficou em 0,95).
5. **App.** A tela em Streamlit recebe a imagem, mostra a probabilidade de pneumonia e avisa quando o resultado está perto do limiar.

## Resultados

Teste com 391 imagens de pacientes que o modelo nunca viu (registro completo em `treino_v2_log.txt`):

| Métrica | Valor |
|---|---|
| Acurácia | 93,1% |
| AUC | 0,991 |
| Sensibilidade (pneumonias encontradas) | 97,3% (181 de 186) |
| Especificidade (normais reconhecidos) | 89,3% (183 de 205) |

Matriz de confusão:

| | Previsto normal | Previsto pneumonia |
|---|---|---|
| **Real normal** | 183 | 22 |
| **Real pneumonia** | 5 | 181 |

O modelo deixou passar 5 pneumonias e deu 22 alarmes falsos. Numa triagem, deixar passar um doente é o erro mais grave, por isso a sensibilidade alta pesa mais do que a acurácia.

## Tecnologias

Python, TensorFlow/Keras (MobileNetV2), scikit-learn (divisão por paciente e métricas), Streamlit, NumPy e Pillow.

## Como rodar

1. Instale o Python 3.10 ou mais novo e as dependências:

   ```
   pip install -r requirements.txt
   ```

2. Para usar o modelo já treinado (`modelo_pneumonia_v2.keras`), abra o app:

   ```
   streamlit run app_pneumonia.py
   ```

3. Para treinar de novo, baixe o banco no [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), coloque as imagens em duas pastas, `NORMAL` e `PNEUMONIA`, ajuste a variável `RAIZ` no início do `treinar_com_minhas_imagens.py` e rode:

   ```
   python treinar_com_minhas_imagens.py
   ```

## Arquivos

| Arquivo | O que faz |
|---|---|
| `treinar_com_minhas_imagens.py` | Versão final: divide por paciente, treina a MobileNetV2, escolhe o limiar e avalia no teste |
| `app_pneumonia.py` | App em Streamlit (upload ou câmera) |
| `modelo_pneumonia_v2.keras` e `limiar.json` | Modelo final e limiar usados pelo app |
| `treino_v2_log.txt` | Saída do treino final, com as métricas acima |
| `treinar_ia_melhorada.py`, `modelo_pneumonia_melhorado.keras`, `treino_log.txt` | Primeira versão: rede convolucional treinada do zero, mantida para comparação |

## Limitações

- As imagens são de crianças de 1 a 5 anos de um único hospital. Nada garante que o modelo funcione com adultos ou com aparelhos de outros lugares.
- O modelo só conhece duas respostas. Diante de qualquer imagem, ele vai responder normal ou pneumonia.
- Fotos de tela ou de filme têm brilho e ângulo diferentes das imagens de treino, e o resultado pela câmera é menos confiável do que pelo upload do arquivo.
- O teste tem 391 imagens do mesmo banco. Um sistema real passa por validação com médicos, em outros hospitais, e por aprovação regulatória.

## Autoria

Projeto individual de Aliny Gomes: preparação dos dados, treino e avaliação do modelo, app e documentação.

## Fonte dos dados

Banco [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), publicado no Kaggle por Paul Mooney, com licença CC BY 4.0. São cerca de 5.860 raio-X de tórax de crianças de 1 a 5 anos atendidas no Guangzhou Women and Children's Medical Center, em Guangzhou, na China.

Dados originais: KERMANY, D.; ZHANG, K.; GOLDBAUM, M. *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*. Mendeley Data, v2, 2018. DOI: 10.17632/rscbjbr9sj.2.

As imagens não estão neste repositório. Baixe direto do Kaggle.
