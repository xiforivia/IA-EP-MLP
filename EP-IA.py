from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.datasets import fashion_mnist
import tensorflow as tf
from keras import models
from keras import layers
from keras import regularizers
from keras.utils import to_categorical
from sklearn.model_selection import KFold
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

file = open('resultado.txt', 'w')
sys.stdout = file

def pre_processamento():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape((60000,28,28,1))
    train_images = train_images.astype('float32')/255 # Modificar os valores de cada pixel para que eles variem de 0 a 1 melhorará a taxa de aprendizado do nosso modelo.

    test_images = test_images.reshape((10000,28,28,1))
    test_images = test_images.astype('float32')/255 # Modificar os valores de cada pixel para que eles variem de 0 a 1 melhorará a taxa de aprendizado do nosso modelo.

    train_labels = to_categorical(train_labels) # Nosso modelo não pode trabalhar com dados categóricos diretamente. Portanto, devemos usar uma codificação quente. Em uma codificação ativa, os dígitos de 0 a 9 são representados como um conjunto de nove zeros e um único. O dígito é determinado pela localização do número 1. Por exemplo, você representaria um 3 como [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    test_labels = to_categorical(test_labels) # one hot encoding

    return train_images, train_labels, test_images, test_labels

# Definir a arquitetura da MLP
def criar_modelo_mlp(num_neuronios, num_camadas):

    camadas = [num_neuronios] * num_camadas # lista contendo a quantidade de neurônios desejada repetida 'num_camadas'

    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for neurons in camadas:
        model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def criar_grafico(df, titulo):
    plt.cla()
    plt.plot(df['acuracia'], 'b', marker='.', label='acurácia', linewidth=3, markersize=12)
    plt.title(titulo)
    plt.legend()
    plt.savefig(titulo+".png", format='png', dpi=300, facecolor='white')

### Main

train_images, train_labels, test_images, test_labels = pre_processamento()


k_folds = 5

cv = KFold(n_splits=k_folds, shuffle=True, random_state=42) # vamos embaralhá-los antes de dividi-lo, seed 42

# --------- Testar diferentes quantidades de camadas e de neurônios ------------
lista_num_camadas = [1, 3]  # Quantidade de neurônios em cada camada
lista_num_neuronios = [25, 100]  # Quantidade de camadas escondidas

dct_camada_neuronio = {}
melhor_acc = 0
melhor_num_camada = 0
melhor_num_neuronio = 0
for num_camadas in lista_num_camadas:
    for num_neuronios in lista_num_neuronios:
        fold_no = 1 #contador
        acc_per_fold = [] #acurácia de cada fold
        for train, test in cv.split(train_images, train_labels): #pra cada fold
            print(f"Treinando fold {fold_no}, com {num_camadas} camada(s) oculta, com {num_neuronios} neurônios" )
            train_X = train_images[train]
            test_X = train_images[test]
            model = criar_modelo_mlp(num_neuronios, num_camadas)
            model.fit(train_X, train_labels[train], epochs=5, batch_size=64, verbose=2)
            test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
            acc_per_fold.append(test_acc * 100)
            fold_no = fold_no + 1

        media_acc_camada_neuronio = sum(acc_per_fold)/len(acc_per_fold)

        dct_camada_neuronio.update({str(num_camadas)+"_"+str(num_neuronios): {"acuracia": media_acc_camada_neuronio}})

        print(f"Média acurácia dos 5 folds pra {num_camadas} camada(s) oculta, com {num_neuronios} neurônios:", media_acc_camada_neuronio)
        print(f"Essa acurácia significa que o modelo usando {num_camadas} camada(s) oculta, com {num_neuronios} neurônios, usando a função ReLu para os neurônios das camadas ocultas e SoftMax para a saída é capaz de classificar corretamente em média {round(media_acc_camada_neuronio, 1)}% das imagens")
        if media_acc_camada_neuronio > melhor_acc:
            melhor_acc = media_acc_camada_neuronio
            melhor_num_camada = num_camadas
            melhor_num_neuronio = num_neuronios
print(f"Portanto, a melhor quantidade de camada(s) oculta é {melhor_num_camada}, com {melhor_num_neuronio} neurônios, que possui {round(melhor_acc, 1)} de acurácia.")
dfCamadaNeuronio = pd.DataFrame(dct_camada_neuronio).T
titulo = "Acurácia de cada qtde de camadas de oculta e qtde de neurônios"
criar_grafico(dfCamadaNeuronio, titulo)


# ----- Taxa de Regularização L2 -----

# A taxa de regularização L2, também conhecida como penalidade L2 ou regularização de peso L2, 
# é uma técnica utilizada durante o treinamento de modelos de aprendizado de máquina para evitar o overfitting,
# que ocorre quando o modelo se torna muito complexo e se ajusta excessivamente aos dados de treinamento, perdendo a capacidade de generalizar para novos dados.

# A taxa de regularização L2 é aplicada adicionando um termo à função de perda durante o treinamento do modelo.
# Esse termo penaliza pesos grandes no modelo, incentivando-os a ter valores menores. Essa penalidade ajuda a evitar
# que os pesos fiquem muito altos, tornando o modelo mais suave e reduzindo a complexidade geral.

def criar_modelo_mlp_regularizacao(num_neuronios, num_camadas, taxa_regularizacao):

    camadas = [num_neuronios] * num_camadas # lista contendo a quantidade de neurônios desejada repetida 'num_camadas'
    
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for neurons in camadas:
        model.add(layers.Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(taxa_regularizacao)))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# --------- Testar diferentes taxas de regularização L2 ------------

taxas_regularizacao = [0.001, 0.01, 0.1]  # Taxas de regularização L2 a serem testadas

dct_taxa_regularizacao = {}
melhor_acc = 0
melhor_taxa_regularizacao = 0
for taxa_regularizacao in taxas_regularizacao:
        fold_no = 1 #contador
        acc_per_fold = [] #acurácia de cada fold
        for train, test in cv.split(train_images, train_labels): #pra cada fold
            print(f"Treinando fold {fold_no}, com {melhor_num_camada} camada(s) oculta, com {melhor_num_neuronio} neurônios e {taxa_regularizacao} de taxa de regularização L2" )
            train_X = train_images[train]
            test_X = train_images[test]
            model = criar_modelo_mlp_regularizacao(melhor_num_neuronio, melhor_num_camada, taxa_regularizacao)
            model.fit(train_X, train_labels[train], epochs=5, batch_size=64, verbose=2)
            test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
            acc_per_fold.append(test_acc * 100)
            fold_no = fold_no + 1

        media_acc_regularizacao = sum(acc_per_fold)/len(acc_per_fold)

        dct_taxa_regularizacao.update({str(taxa_regularizacao): {"acuracia": media_acc_regularizacao}})

        print(f"Média acurácia dos 5 folds pra {melhor_num_camada} camada(s) oculta, com {melhor_num_neuronio} neurônios e {taxa_regularizacao} de taxa de regularização L2:", media_acc_regularizacao)
        print(f"Essa acurácia significa que o modelo usando {melhor_num_camada} camada(s) oculta, com {melhor_num_neuronio} neurônios, com {taxa_regularizacao} de taxa de regularização L2 e usando a função ReLu para os neurônios das camadas ocultas e SoftMax para a saída é capaz de classificar corretamente em média {round(media_acc_regularizacao, 1)}% das imagens")
        if media_acc_regularizacao > melhor_acc:
            melhor_acc = media_acc_regularizacao
            melhor_taxa_regularizacao = taxa_regularizacao
print(f"Portanto, a melhor taxa de regularização L2 é {taxa_regularizacao}, que possui {round(melhor_acc, 1)} de acurácia.")
dfTaxaRegularizacao = pd.DataFrame(dct_taxa_regularizacao).T
titulo = "Acurácia de cada taxa de regularização L2"
criar_grafico(dfTaxaRegularizacao, titulo)


# --------- Testar diferentes quantidades de feature maps (filters) ------------

filters_list = [16, 32, 64]

dct_filters = {}
melhor_acc = 0
melhor_filter = 0
for filters in filters_list:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    for train, test in cv.split(train_images, train_labels):
        print(f"Treinando fold {fold_no}, com {filters} filters" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo(melhor_layer, filters, dense_size)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
        acc_per_fold.append(test_acc * 100)
        fold_no = fold_no + 1

    media_acc_filters = sum(acc_per_fold)/len(acc_per_fold)

    dct_filters.update({str(filters): {"acuracia": media_acc_filters}})

    print(f"Média acurácia dos 5 folds pra {filters} filters:", media_acc_filters)
    print(f"Essa acurácia significa que o modelo usando {melhor_layer} camada(s) de convolução-pooling, filter: {filters} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente em média {round(media_acc_filters, 1)}% das imagens")
    if media_acc_filters > melhor_acc:
        melhor_acc = media_acc_filters
        melhor_filter = filters
print(f"Portanto, a melhor quantidade de filters é {melhor_filter}, que possui {round(melhor_acc, 1)} de acurácia.")
dfFilter = pd.DataFrame(dct_filters).T
titulo = "Acurácia de cada qtde de feature maps"
criar_grafico(dfFilter, titulo)

# --------- Testar diferentes tamanhos de camada densa ------------

dense_size_list = [64, 128, 256]

dct_dense = {}
melhor_acc = 0
melhor_dense = 0
for dense_size in dense_size_list:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    for train, test in cv.split(train_images, train_labels): #pra cada fold
        print(f"Treinando fold {fold_no}, com tamanho da camada densa: {dense_size}" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo(melhor_layer, melhor_filter, dense_size)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
        acc_per_fold.append(test_acc * 100)
        fold_no = fold_no + 1

    media_acc_dense = sum(acc_per_fold)/len(acc_per_fold)

    dct_dense.update({str(dense_size): {"acuracia": media_acc_dense}})

    print(f"Média acurácia dos 5 folds pra tamanho da camada densa: {dense_size}:", media_acc_dense)
    print(f"Essa acurácia significa que o modelo usando {melhor_layer} camada(s) de convolução-pooling, filter: {melhor_filter} e tamanho da camada densa: {dense_size} é capaz de classificar corretamente em média {round(media_acc_dense, 1)}% das imagens")
    if media_acc_dense > melhor_acc:
        melhor_acc = media_acc_dense
        melhor_dense = dense_size
print(f"Portanto, o melhor tamanho da camada densa é {melhor_dense}, que possui {round(melhor_acc, 1)} de acurácia.")
dfDense = pd.DataFrame(dct_dense).T
titulo = "Acurácia pra cada quantidade de neurônios da camada densa"
criar_grafico(dfDense, titulo)

# Resumo
print(f"Ao final, a melhor combinação foi: \n{melhor_layer} camada(s) de convolução-pooling, {melhor_filter} filter(s) e tamanho da camada densa: {melhor_dense}")

# ----- Dropout -----

# O Dropout é uma técnica de regularização utilizada para reduzir o overfitting em redes neurais.
# Durante o treinamento, uma proporção dos neurônios é aleatoriamente "desligada" (dropout) em cada atualização do gradiente,
# o que força a rede a aprender recursos mais robustos e evita a dependência excessiva de neurônios específicos.

# Vamos modificar a função create_cnn_model_with_dense_size para adicionar uma camada Dropout antes da camada densa:
def criar_modelo_com_dropout(conv_layers, filters, dense_size, dropout_rate):
    model = models.Sequential()

    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        print("MaxPooling2D")

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.Dropout(dropout_rate)) #Dropout adicionado
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

# --------- Testar diferentes porcentagens de dropout ------------

dropout_rates = [0.1, 0.3, 0.5, 0.7]

dct_dropout = {}
melhor_acc = 0
melhor_dropout = 0
for dropout_rate in dropout_rates:
    fold_no = 1 #contador
    acc_per_fold = [] #acurácia de cada fold
    for train, test in cv.split(train_images, train_labels): #pra cada fold
        print(f"Treinando fold {fold_no}, com {dropout_rate} de dropout_rate" )
        train_X = train_images[train]
        test_X = train_images[test]
        model = criar_modelo_com_dropout(melhor_layer, melhor_filter, melhor_dense, dropout_rate)
        model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
        test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2, )
        acc_per_fold.append(test_acc * 100)
        fold_no = fold_no + 1

    media_acc_dropout = sum(acc_per_fold)/len(acc_per_fold)

    dct_dropout.update({str(dropout_rate): {"acuracia": media_acc_dropout}})

    print(f"Média acurácia dos 5 folds pra {dropout_rate} de dropout_rate:", media_acc_dropout)
    print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense} e dropout: {dropout_rate} é capaz de classificar corretamente em média {round(media_acc_dropout, 1)}% das imagens")
    if media_acc_dropout > melhor_acc:
        melhor_acc = media_acc_dropout
        melhor_dropout = dropout_rate
print(f"Portanto, o melhor dropout é {melhor_dropout}, que possui {round(melhor_acc, 1)} de acurácia.")
dfDropout = pd.DataFrame(dct_dropout).T
titulo = "Acurácia de cada porcentagem de dropout"
criar_grafico(dfDropout, titulo)

# ----- Batch Normalization -----

# Batch Normalization é uma técnica usada para acelerar o treinamento de redes neurais e estabilizar o processo de aprendizado.
# Ela normaliza a ativação de cada camada, aplicando uma transformação que mantém a média próxima de zero e o desvio padrão próximo de um.
# Isso ajuda a reduzir a covariância de ativação entre as camadas e torna o treinamento mais rápido e estável.

# Vamos modificar a função create_cnn_model_with_dense_dropout para adicionar uma camada Batch Normalization antes da camada densa:
def criar_modelo_com_batchnorm(conv_layers, filters, dense_size, dropout_rate):
    model = models.Sequential()

    for i in range(conv_layers):
        model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape = (28,28,1)))
        print("Conv2D")
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        print("MaxPooling2D")

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_size, activation='relu'))
    model.add(layers.BatchNormalization()) # Batch Normalization adicionado
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(10, activation='softmax')) # 10 é o número de classes
    
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

dct_batchnorm = {}
fold_no = 1 #contador
acc_per_fold = [] #acurácia de cada fol
for train, test in cv.split(train_images, train_labels): #pra cada fold
    print(f"Treinando fold {fold_no}, com batch normalization" )
    train_X = train_images[train]
    test_X = train_images[test]
    model = criar_modelo_com_batchnorm(melhor_layer, melhor_filter, melhor_dense, melhor_dropout)
    model.fit(train_X, train_labels[train], epochs=epochs, batch_size=batch_size, verbose=2)
    test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
    acc_per_fold.append(test_acc * 100)
    fold_no = fold_no + 1

media_acc_batchnorm = sum(acc_per_fold)/len(acc_per_fold)

dct_batchnorm.update({"acuracia": media_acc_batchnorm})

print(f"Média acurácia dos 5 folds com batch normalization:", media_acc_batchnorm)
print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense}, dropout: {melhor_dropout} e com batch normalization é capaz de classificar corretamente em média {round(media_acc_batchnorm, 1)}% das imagens")

print(f"Portanto, possui em média {round(media_acc_batchnorm, 1)} de acurácia com o batch normalization.")

# ----- Data Augmentation -----

# Data Augmentation é uma técnica usada para expandir o conjunto de dados de treinamento, aplicando transformações aleatórias nos dados existentes,
# como rotação, zoom, espelhamento, deslocamento, entre outros. Essa técnica é útil quando o conjunto de dados de treinamento é limitado,
# pois permite aumentar a diversidade dos exemplos apresentados ao modelo.


# Criar um gerador de data augmentation
augmenter = ImageDataGenerator(
    rotation_range=20, # podem ser rotacionadas aleatoriamente em um ângulo de -20 a 20 graus
    width_shift_range=0.2, # as imagens podem ser deslocadas horizontalmente em até 20% da largura da imagem
    height_shift_range=0.2,  # as imagens podem ser deslocadas verticalmente em até 20% da largura da imagem
    shear_range=0.2, # as imagens podem ser distorcidas com um valor de cisalhamento aleatório entre -0.2 e 0.2
    zoom_range=0.2, # as imagens podem ser ampliadas ou reduzidas em até 20% aleatoriamente.
    horizontal_flip=True # imagens podem ser invertidas horizontalmente durante o data augmentation.
)

# Ajustar o gerador aos dados de treinamento
# datagen.fit(train_images.reshape((-1, 28, 28, 1)))

dct_dataaug = {}
fold_no = 1 #contador
acc_per_fold = [] #acurácia de cada fol #loss de cada fold
for train, test in cv.split(train_images, train_labels): #pra cada fold
    print(f"Treinando fold {fold_no}, com data augmentation")
    
    # Obter os conjuntos de treinamento e teste para o fold atual
    train_X, test_X = train_images[train], train_images[test]
    train_y, test_y = train_labels[train], train_labels[test]
    
    # Aplicar o data augmentation aos dados de treinamento
    augmenter.fit(train_X)
    augmented_train_X = augmenter.flow(train_X, train_y, batch_size=32)

    model = criar_modelo_com_batchnorm(melhor_layer, melhor_filter, melhor_dense, melhor_dropout)

    model.fit(augmented_train_X, epochs=5, validation_data=(test_X, test_y), verbose=2)

    test_loss, test_acc = model.evaluate(test_X, train_labels[test], verbose=2)
    acc_per_fold.append(test_acc * 100)
    fold_no = fold_no + 1

media_acc_dataaug = sum(acc_per_fold)/len(acc_per_fold)

dct_dataaug.update({"acuracia": media_acc_dataaug})

print(f"Média acurácia dos 5 folds com data augmentation:", media_acc_dataaug)
print(f"Essa acurácia significa que o modelo usando layer: {melhor_layer}, filter: {melhor_filter}, tamanho da camada densa: {melhor_dense}, dropout: {melhor_dropout}, com batch normalization e com data augmentation é capaz de classificar corretamente em média {round(media_acc_dataaug, 1)}% das imagens")

print(f"Portanto, possui em média {round(media_acc_dataaug, 1)} de acurácia.")


d = {'acuracia':[dct_layer[str(melhor_layer)]['acuracia'], dct_filters[str(melhor_filter)]['acuracia'], dct_dense[str(melhor_dense)]['acuracia'], dct_dropout[str(melhor_dropout)]['acuracia'], dct_batchnorm['acuracia'], dct_dataaug['acuracia']]}
print(d)
dfFinal = pd.DataFrame(data=d, index=['convolução-pooling','feature maps', 'camada densa', 'dropout', 'batch normalization', 'data augmentation'])

plt.cla()
plt.figure(figsize=(10,10)) 
plt.plot(dfFinal, 'b', marker='.', label='Acurácia', linewidth=3, markersize=12)
titulo = "Mudança na Acurácia após cada modificação"
plt.title(titulo)
plt.legend(loc='upper left')
plt.savefig(titulo+".png", format='png', dpi=300, facecolor='white', bbox_inches='tight')


sys.stdout = sys.__stdout__
file.close()