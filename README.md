### O projeto consiste em aplicar o método de Transfer Learning em uma rede de Deep Learning na linguagem Python no ambiente COLAB.
---

# **Introdução: Transfer Learning**

**Transfer learning** (ou aprendizado por transferência) é o processo de reutilizar um modelo previamente treinado em um conjunto de dados grande para resolver um problema diferente, geralmente com um conjunto de dados menor. Ele envolve:

- **Extração de Recursos (Feature Extraction)**: Utiliza um modelo pré-treinado como um extrator de características, aproveitando os pesos congelados de camadas iniciais do modelo.
- **Fine-Tuning**: Ajusta os pesos de algumas camadas, geralmente as últimas, para adaptar o modelo ao novo conjunto de dados.

O notebook foca no uso de transfer learning com o modelo **VGG16**, vencedor do desafio ImageNet de 2014, aplicando essas técnicas em um conjunto de dados de imagens.

---

# **Estrutura do Notebook**

## **1. Preparação do Dataset**
- Processa as imagens para o formato necessário (RGB, 224x224) e divide os dados em:
  - **Treino (70%)**
  - **Validação (15%)**
  - **Teste (15%)**

---

## **2. Visualização de Dados**
- Exibe exemplos de imagens do dataset para inspeção visual.
- Demonstra a distribuição das classes para verificar o equilíbrio.

---

## **3. Comparação entre Modelos**
- Treina um modelo pequeno do zero para comparação.
- Em seguida, aplica transferência de aprendizado com **VGG16**.

---

## **4. Transfer Learning com VGG16**
- Remove a camada final do modelo pré-treinado (classificação para 1000 classes do ImageNet).
- Adiciona uma nova camada de saída para classificar as 2 classes do novo dataset.
- Congela as camadas iniciais da VGG16 para usar como extrator de características.

---

## **5. Fine-Tuning**
- Ajusta os pesos de camadas finais para adaptar o modelo ao novo conjunto de dados.
- Define uma **taxa de aprendizado menor** para preservar as características aprendidas.

---

## **6. Treinamento e Avaliação**
- Usa otimizadores e métricas como **Adam** e **accuracy**.
- Avalia a performance no conjunto de validação durante o treinamento.
- Testa o modelo final no conjunto de teste, comparando com o treinamento do modelo do zero.

---

# **Vantagens Demonstradas**
- Aumenta a acurácia com poucos dados, aproveitando recursos genéricos aprendidos por **VGG16**.
- Reduz o tempo de treinamento em comparação com treinar um modelo do zero.
- Mostra que **fine-tuning** melhora a performance em tarefas específicas.

---

# **Resultados e Conclusão**
- O notebook ilustra como a técnica de transfer learning pode superar a abordagem tradicional, fornecendo resultados expressivos mesmo com datasets limitados.
- Também oferece flexibilidade para aplicar estratégias diferentes, dependendo das características dos dados.

---

### Explicação Detalhada do Código

O código implementa **Transfer Learning** usando o modelo pré-treinado **VGG16** e compara sua performance com um modelo básico treinado do zero. Aqui está a explicação detalhada de cada etapa:

---

## **1. Introdução: O que é Transfer Learning?**
O objetivo do Transfer Learning é reutilizar características aprendidas por um modelo em um grande conjunto de dados (como o ImageNet) para treinar em um conjunto menor, economizando tempo e aumentando a acurácia. Isso é feito de duas formas:
- **Extração de Características**: Congela camadas iniciais do modelo para reutilizar seus pesos.
- **Fine-Tuning**: Ajusta as camadas finais para se adaptar ao novo dataset.

---

## **2. Preparação dos Dados**

### **2.1 Download e Organização**
O código baixa o dataset **kagglecatsanddogs**, que contém imagens organizadas em pastas por classe. Ele processa e organiza as imagens, mantendo apenas arquivos `.jpg`.

#### Código:
```bash
!echo "Downloading kagglecatsanddogs for image notebooks"
!curl -L -o kagglecatsanddogs_5340.zip --progress-bar https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip
!unzip kagglecatsanddogs_5340.zip # Extract the outer .zip
!rm kagglecatsanddogs_5340.zip
!ls PetImages # List the contents of the desired folder
```

Esse comando baixa e descompacta o dataset. 

### **2.2 Pré-processamento**
Cada imagem é redimensionada para 224x224 pixels, formato esperado pela **VGG16**:
```python
def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x
```
- `load_img`: Carrega a imagem.
- `target_size`: Redimensiona para 224x224.
- `preprocess_input`: Normaliza os dados de entrada.

Além disso, os dados são divididos em:
- **70% treino**
- **15% validação**
- **15% teste**

### **2.3 Normalização e Codificação**
Os valores dos pixels são normalizados para `[0, 1]` para melhorar o desempenho do modelo:
```python
x_train = x_train.astype('float32') / 255.
x_val = x_val.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
```
As classes são transformadas em vetores **one-hot**:
```python
y_train = keras.utils.to_categorical(y_train, num_classes)
```

---

## **3. Construção de um Modelo Base**
O modelo inicial é simples e treinado do zero para fins de comparação:
```python
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])
```

### Explicação:
- **Camadas Convolucionais (`Conv2D`)**: Extraem características das imagens, como bordas e texturas.
- **Camadas de Pooling (`MaxPooling2D`)**: Reduzem a dimensionalidade dos mapas de características.
- **Camada Densa (`Dense`)**: Realiza a classificação final.
- **Função de Ativação (`relu`, `softmax`)**:
  - `relu`: Ativação linear retificada, usada em camadas ocultas.
  - `softmax`: Gera probabilidades para cada classe.

### Treinamento:
```python
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```
- **Otimizador**: `adam` ajusta os pesos para minimizar a função de perda.
- **Função de perda**: `categorical_crossentropy` mede a diferença entre as predições e os rótulos reais.

**Resultado**: ~49% de acurácia devido ao pequeno dataset.

---

## **4. Transfer Learning com VGG16**

### **4.1 Carregamento do Modelo**
O modelo **VGG16** pré-treinado é carregado:
```python
vgg = keras.applications.VGG16(weights='imagenet', include_top=False)
```
- **`weights='imagenet'`**: Usa pesos treinados no dataset ImageNet.
- **`include_top=False`**: Remove as camadas densas finais para personalização.

### **4.2 Modificação da Arquitetura**
Adiciona-se uma nova camada de saída para classificar 2 classes:
```python
x = vgg.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model_new = Model(inputs=vgg.input, outputs=predictions)
```

### **4.3 Congelamento de Pesos**
Congela as camadas iniciais para preservar os pesos aprendidos no ImageNet:
```python
for layer in vgg.layers:
    layer.trainable = False
```

### **4.4 Treinamento**
A nova camada é treinada:
```python
model_new.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history2 = model_new.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
```
Resultado: ~80% de acurácia com Transfer Learning.

---

## **5. Fine-Tuning**
Descongela as camadas finais para ajustar o modelo ao novo dataset:
```python
for layer in vgg.layers[-4:]:
    layer.trainable = True
model_new.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
```
- Apenas as últimas 4 camadas são ajustadas.
- A taxa de aprendizado é reduzida para evitar grandes alterações nos pesos.

---

## **6. Resultados**
- **Modelo Inicial**: ~49% de acurácia.
- **Transfer Learning (sem Fine-Tuning)**: ~80% de acurácia.
- **Transfer Learning com Fine-Tuning**: Potencial para ultrapassar 80%, dependendo dos ajustes.

---

### **Conclusão**
O código demonstra a eficiência do Transfer Learning:
- **Economia de Tempo**: Treinamento mais rápido.
- **Aumento da Acurácia**: Melhores resultados em comparação ao modelo inicial.
- **Flexibilidade**: Possibilidade de aplicar Fine-Tuning em casos específicos.


---
---

# Referências

https://www.tensorflow.org/datasets/catalog/cats_vs_dogs?hl=pt-br

https://www.microsoft.com/en-us/download/details.aspx?id=54765

