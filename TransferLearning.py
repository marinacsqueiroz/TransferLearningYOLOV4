# %% [markdown]
# # Donwload da rede pré treinada YoloV4

# %% [markdown]
# Para baixar e ajustar os arquivos do darknet para treinamento da rede, foram utilizados os tutorias presentes em:
#    
# 
# *    https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
# *   https://www.youtube.com/watch?v=BIvEnrlliBY
# 
# *   https://www.youtube.com/watch?v=sKDysNtnhJ4
# 
# 
# 
# 
# 

# %%
from google.colab import drive
import os
drive.mount('/content/drive')

# %%
path = '/content/drive/MyDrive'
os.chdir(path)

# %%
!git  clone https://github.com/AlexeyAB/darknet

# %%
!/usr/local/cuda/bin/nvcc --version

# %% [markdown]
# Os passos acima foram feitos para baixar os arquivos necessários, em seguida foram feitas as modificações nos arquivos segundo os links citados. Foi feito o download de pesos da rede yolov4.conv.137, disponível no darknet.
# 
# Para permitir que o programa rode sem necessidade de fazê-los novamente, a pasta darknet foi baixada para o computador e assim realizadas as modificações assim como as inserções das imagens a serem utilizadas para treinamento.
# 
# No arquivo cfg utilizado também é possível escolher a iou_loss a ser utilizada. Foi mantida a original do arquivo, CIoU , esta pode minimizar diretamente a distância entre dois alvos, o que cria uma converção rápida, esse método leva em consideração o grandiente.
# 
# As modificações feitas foram:
#   1. Criação da copia do arquivo yolov4-custom.cfg e modificação para o arquivo yolo-obj.cfg.
#   2. Alterações feitas:        
#     *   batch = 64
# 
#     *   subdivisions = 16
# 
#     *   max_batches = 6000
# 
#     *   steps = 4800,5400
#    
#     *   width = 416
# 
#     *   height = 416
# 
#     *   Redes convolucionais anteriores as yolo:
#           filters = 24 
# 
#     *   Redes yolo:
#           classes = 3
# 
# 
# 
#              
#         
#   
#       
#       

# %% [markdown]
# A seguir é modificado o arquivo Makefile para uso da GPU e OpenCV e realizar o 'build' no darknet

# %%
%cd darknet/
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
!sed -i 's/LIBSO=0/LIBSO=1/' Makefile

# %%
!make

# %%
!./darknet

# %% [markdown]
# # Processamento das imagens

# %% [markdown]
# Para o treinamento foram retiradas 700 imagens de cada classe (70%), para que fosse possível fazer a avaliação do treinamento com imagens testes não vistas pelo modelo. Adicionar quantidades balanceadas de cada classe também é uma boa prática uma vez que um banco de dados balanceado fornece resultados mais satisfatórios.
# 
# Para criar os arquivos textos citados nos links, para cada imagem de treinamento foram feitos por meio do programa disponível em https://github.com/heartexlabs/labelImg/releases/tag/v1.8.1.
# 
# Essas imagens já estão disponíveis no data do arquivo darknet presente neste trabalho, no arquivo data/obj. As caixas delimitadoras das imagens foram feitas em volta das folhas, esperando que a rede detectasse as moscas brancas e entendesse que sua quantidade deveria ser utilizada como preditor.
# 

# %% [markdown]
# # Treinamento

# %%

# primeiro treinamento
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg yolov4.comv.137 -dont_show -map
# primeiro treinamento
!./darknet detector train data/obj.data cfg/yolov4-obj.cfg backup/yolov4-obj_last.weights -dont_show -map


# %% [markdown]
# Por falta de espaço no google drive, o treinamento foi feito até o batch 1910, com a útilma medida das métricas de acurácias medidas no batch 1886.
# 
# O treinamento foi feito com diversas passadas, por causa dessa limitação. Seus pesos eram salvos na pasta backup e assim foi possível dar continuidade aos treinos, selecionando os pesos a serem observados (backup/yolov4-obj_last.weights). 
# 
# Para a primeira passada de treino foi utilizada os pesos de yolov4.conv.137.

# %% [markdown]
# # Teste

# %% [markdown]
# Foram separadas no início do treinamento, 900 imagens , 300 de cada classe para avaliar o treinamento. O próprio programa realiza de tempos em tempos as métricas conseguidas pelo treinamento, este é feito em cima das imagens separadas para validação, ou seja, não são usadas no treinamento então podemos nos basear por essas métricas encontradas.
# 
# Outro ponto é que se faz necessário o uso deve arquivo de validação para utilizar essas métricas, caso contrário elas seriam feitas em cima das imagens de treinamento, o que causaria um viés uma vez que a rede aprenderia e faria testes com a mesma imagem.
# 
# A seguir tem-se uma comparação a ser feita, com a primeira e última métrica medida (1000 e 1886). A seguir tem-se a legenda dos resultados
# 
# TP = Verdadeiro positivo
# 
# FP = Falso positivo
# 
# FN = Falso negativo
# 
# ap = Average precision 
# 
# Mape 50% = Mean Absolute Percentage Error
# 
# average IoU = é uma métrica que calcula quantos porcento as caixas preditoras acertaram as caixas verdadeiras da imagem. Quanto maior a porcentagem, mais próximas, um resultado considerado bom é acima de 50%.
# 
# Para cada classe foram calculadas a precisão de cada classe (TP/TP+FP), Falsos negativos (valor total de imagens daquela classe - TP), Recall (TP/TP+FN) e F1(2 * precision * recall/(precision+recall))
# 
# ---
# Batch 1000:
# ---
# ---
# 
# *   class_id = 0, name = low_abundance, ap = 90.69%   	 
#   (TP = 161, FP = 4, FN = 539)
#   precision = 0.97
#   recall = 0.23
#   F1-score = 0.37
# *   class_id = 1, name = moderate_abundance, ap = 26.69%   	
#   (TP = 78, FP = 272, FN = 622) 
#   precision = 0.22
#   recall = 0.11
#   F1-score = 0.14
# *   class_id = 2, name = super_abundance, ap = 46.17%   	 
#   (TP = 246, FP = 289, FN = 454)
#   precision = 0.45
#   recall = 0.35
#   F1-score = 0.39
# 
# 
# *   precision = 0.46
# *   recall = 0.54
# *   F1-score = 0.50 
# *   TP = 485
# *   FP = 565
# *   FN = 415
# *   average IoU = 34.31 % 
# *   mean average precision (mAP@0.50) = 0.545177, or 54.52 % 
# 
# IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
# 
# 
# ---
# Batch 1886:
# ---
# ---
# 
# *   class_id = 0, name = low_abundance, ap = 92.96%   	 
#   (TP = 182, FP = 7, FN = 518)
#   precision = 0.96
#   recall = 0.26
#   F1-score = 0.79
# *   class_id = 1, name = moderate_abundance, ap = 26.69%   	
#   (TP = 160, FP = 355, FN = 540)
#   precision = 0.31
#   recall = 0.22
#   F1-score = 0.25
# *   class_id = 2, name = super_abundance, ap = 46.17%   	 
#   (TP = 186, FP = 222, FN = 514)
#   precision = 0.45
#   recall = 0.26
#   F1-score = 0.32
# 
# *   precision = 0.47
# *   recall = 0.59
# *   F1-score = 0.52 
# *   TP = 528
# *   FP = 584
# *   FN = 372
# *   average IoU = 37.75 % 
# *   mean average precision (mAP@0.50) = 0.579188, or 57.92 % 
# 
# IoU threshold = 50 %, used Area-Under-Curve for each unique Recall 
# 
# O valor total de loss foi de 1.386494, com o avg loss error de 1.607449. O learning rate utilizado foi mantido ao original do arquivo cfg copiado (0.001000). Com um tamanho batch de 64, e uma subdivisão de 16, durante o treinamento as interações foram feitas em grupos de 16 em 16 imagens. Por útilmo foram vistas 1207704 imagens no treinamento (64*1886). 
# 
# Last accuracy mAP@0.50 = 63.36 %, best = 63.36 %
# 
# 
# 
# 

# %% [markdown]
# # Avaliação dos resultados

# %% [markdown]
# A classe de moderada abundância apresentou os piores resultados do início ao fim, e os melhores ocorreram na baixa abundância.
# 
# Pode-se entender que o modelo até onde foi feito o treinamento não conseguiu compreender bem como deveria fazer o padrão para classificação, isso porque a melhoria de métricas cálculadas nos batch 1000 e 1886 foram mínimas.
# 
# O modelo apresentou o melhor MAPE de 63.36% e um average IoU de 37.75 %. Mesmo apresentando uma média de acurácia de 63%, obteve um IoU menor que 50% o que demonstra que o modelo não é satisfatório

# %% [markdown]
# # Possível melhoramento de métricas

# %% [markdown]
# Segundo https://github.com/AlexeyAB/darknet#how-to-improve-object-detection, há algumas maneiras de melhorar o treinamento e precisão do modelo. 
# 
# Além desses pontos, para esse caso especifico, não foram utilizados nem metade dos batch que deveriam ser feitos (metade = 3000), por falta de memória do google colab, então realizadar esse treinamento em um colab pro ou em computador local pode gerar métricas melhores. 
# 

# %% [markdown]
# # Comparação com Faster-RCNN

# %% [markdown]
# As imagens treinadas pelo modelo RCNN foram classificadas da mesma forma, porém as caixas feitas nas imagens foram feitas em casa mosca branca, não da folha inteira como feito na transferência de treinamento da yolo v4.
# 
# O RCNN teve uma precisão de 0.98 e o yolo de 0.47. Já o recall do RCNN foi de 0.81 contra 0.59 do yolo. Por último o F1 do RCNN foi de 0.89 e do yolo 0.52.
# 
# A rede RCNN apresentou resultados melhores que a yolo, esta foi feita em cima de um modelo pré treinado do ResNet, e realizou o treinamento completo, contra o da yolo que não ocorreu nem metade dos batch.
# 
# Outro ponto que pode dar diferença no resultado foi na forma de fazer as labels das imagens, o RCNN criou caixas delimitadores em volta de todas as moscas brancas, enquanto para Yolo foram feitas caixas delimitadores em volta da folha.


