# Explicação do código 

--- 

## 1. Implementando rede neural  

```python
import segmentation_models_pytorch as smp

modelo = smp.Unet(
    encoder_name="mobilenet_v2",        
    encoder_weights="imagenet",         
    in_channels=3,                      
    classes=1                           
)
```

Importação da biblioteca já com os algoritmos construídos.

```python
import segmentation_models_pytorch as smp
```

O PyTorch puro não vem com redes de segmentação prontas e modulares. O `smp` é uma biblioteca construída por cima do PyTorch. Em vez de você escrever manualmente as classes `DoubleConv`, `DownSample` e `UpSample`, o **SMP já tem todas essas fórmulas matemáticas validadas e otimizadas**. Ele cria a estrutura completa da U-Net e gerencia as conexões automáticas entre o lado esquerdo (Encoder) e o direito (Decoder).

```python
modelo = smp.Unet(
    ...
)
```

Você está alocando espaço na memória do computador para milhões de parâmetros (pesos matemáticos e matrizes). A genialidade dessa função específica é que ela é híbrida: **ela constrói o formato clássico em "U", mas permite que você troque o "motor" do lado esquerdo**.


```python
encoder_name="mobilenet_v2", 
```

Ele **substitui as convoluções clássicas pesadas** do lado esquerdo da U-Net (o caminho de contração) pelas Convoluções Separáveis da MobileNetV2.

**Em vez de usar matrizes 3D gigantes para procurar bordas e cores ao mesmo tempo, ele separa o processo, reduzindo a quantidade de multiplicações por frame de bilhões para milhões**.

É isso que **garante que a latência do seu sistema fique baixa o suficiente para rodar em tempo real** na CPU da indústria sem precisar de uma placa de vídeo cara.

```python
encoder_weights="imagenet",
```

**Preenche os "neurônios" do Encoder (MobileNetV2)** com pesos matemáticos que foram treinados para classificar milhões de imagens.

**Se você colocasse None aqui, a sua rede nasceria "cega" e as matrizes começariam com números aleatórios**. Usando imagenet, os filtros convolucionais já nascem sabendo o que é contraste, onde há uma borda reta, o que é um círculo e como a luz se comporta. **O seu modelo só precisa aprender a focar isso na sua chama**.

```python
in_channels=3,
```

**Câmeras digitais padrão formam imagens misturando 3 canais de cor**: Vermelho, Verde e Azul (RGB). Esse parâmetro cria exatamente 3 portas de entrada na primeira camada da rede. Nota de engenharia: **Se no futuro a indústria decidir usar uma Câmera Térmica em preto e branco (infravermelho), você mudaria esse número para 1**.

```python
classes=1
```

Controla a última camada convolucional do Decoder (o lado direito do "U"). A imagem entrou pequena (1 canal), cresceu para 3 canais, passou por convoluções profundas que chegaram a ter 1024 canais no Bottleneck, e foi reconstruída e esticada de volta para a resolução original. **Essa última camada pega os canais restantes e os "esmaga" matematicamente em um único canal** (uma única matriz 2D).

O tensor de saída terá uma camada onde o **valor de cada pixel representará a pontuação bruta (Logits) da rede indicando se aquele pixel pertence à classe alvo (Chama/Copo) ou ao fundo**.

---

## 2. Preparação da classe do dataset 

```python
DIR_TREINO = '.../dataset/train'
```

Utilize o diretório especializado para treino. No caso desse código, as imagens e as imagens de máscaras estava misturadas, com a única diferença de um '_mask' no final do nome do arquivo. 

```python
class Dataset(Dataset):
    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.images = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]
```

**1.** Em vez de carregar as imagens na memória (o que ocuparia toda a sua RAM), ele cria apenas uma lista de nomes de arquivos.

**2.** `if f.endswith('.jpg')` Como o seu dataset mistura imagens, máscaras e arquivos CSV na mesma pasta, essa linha garante que a lista self.images contenha apenas as fotos originais. Sem isso, a rede tentaria "treinar usando uma máscara como se fosse uma foto", o que causaria um erro matemático.

```python
def __len__(self):
    return len(self.images)
```

O PyTorch usa isso para calcular quantas iterações (passos) ele precisa dar para completar uma Época. Se você tem 1545 imagens e um `batch_size` de 4, o PyTorch consulta o '__len__' para saber que precisará de 387 lotes para terminar um ciclo de treino.

```python
def __getitem__(self, idx):
    img_name = self.images[idx]
    mask_name = img_name.replace('.jpg', '_mask.png')
    img_path = os.path.join(self.split_dir, img_name)
    mask_path = os.path.join(self.split_dir, mask_name)
```
**1.** `img_name`: Pega o nome da foto original (ex: caneca123.jpg).

**2.** `mask_name`: O código transforma caneca123.jpg em caneca123_mask.png de forma automática. Isso garante que a rede sempre receba a máscara correta para a imagem correspondente.

```python
image = Image.open(img_path).convert("RGB")
mask = Image.open(mask_path).convert("L")
```

**1.** `.convert("RGB")` garante que, mesmo que uma foto tenha sido salva em escala de cinza ou tenha um canal alfa (transparência RGBA), ela seja forçada a ter exatamente 3 canais. Isso evita que a sua rede (que espera `in_channels=3`) quebre (crash) no meio do treino.

**2.** `.convert("L")` transforma a máscara em uma matriz de um único canal (escala de cinza), já que o gabarito só precisa de valores de intensidade, não de cores.

```python
image = TF.resize(image, (224, 224))
mask = TF.resize(mask, (224, 224), interpolation=Image.NEAREST)
```

**1.** Esmaga ou estica as imagens para o quadrado perfeito que a MobileNetV2 exige.

**2.** Mas você nunca pode fazer isso com a máscara. Se o fundo é 0 e o copo é 1, e o algoritmo "misturar" a borda para suavizar, ele vai criar pixels com valor 0.5. A sua rede entraria em pânico tentando entender o que é a classe 0.5. O `Image.NEAREST` proíbe essa mistura, forçando o pixel novo a ser estritamente igual ao seu vizinho mais próximo. Ele mantém as bordas duras e exatas.

```python
image = TF.to_tensor(image)
mask = TF.to_tensor(mask)
```

Transforma a imagem (que até agora era um objeto da biblioteca PIL) na estrutura de dados nativa do PyTorch (Tensor). Bibliotecas de imagem normais leem imagens no formato `[Altura, Largura, Canais]`. As Redes Convolucionais do PyTorch exigem `[Canais, Altura, Largura]`. O to_tensor faz esse giro automaticamente.

```python
mask = torch.where(mask > 0, 1.0, 0.0)
```
Como você estava usando um dataset multiclasse (vários tipos de copos representados por valores de pixel diferentes), essa linha unifica tudo. Ela garante que a máscara entregue para a sua U-Net seja estritamente binária (Problema de 1 Classe). **Se no futuro você treinar a rede para identificar a chama e a fumaça separadamente, essa é a linha que você vai apagar.**

```python
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

**1.** O `batch_size=4` significa que a rede vai olhar 4 imagens, calcular a média dos erros dessas 4, e só então dar um passo para atualizar seus pesos (Otimização). Isso deixa o aprendizado muito mais estável do que aprender imagem por imagem.

**2.** O `shuffle=True (Embaralhar)` **é a garantia de que a rede não vai decorar a ordem das fotos**. Se todas as imagens escuras estiverem no final da pasta, a rede desaprenderia as claras. Embaralhar garante que todo lote de 4 imagens seja diverso e force o modelo a generalizar.

---

## 3. Treinamento do modelo 

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"O modelo vai treinar usando: {device}")
```

Verifica se tem alguma placa de vídeo na máquina. Se tiver, usa ela, se não, utiliza a CPU.

```python
modelo = modelo.to(device)
```

Move o modelo para a memória da GPU/CPU


```python
criterio = nn.BCEWithLogitsLoss()
```

**É o cálculo de erro padrão para binarização (1 objeto vs Fundo).** Como seu problema é binário (Pixel é Objeto ou Fundo), esta função compara o mapa de probabilidade gerado pela rede com o gabarito real. O "WithLogits" significa que ela aceita os números crus da rede e aplica a função Sigmoid internamente, o que é numericamente mais estável.

```python
otimizador = optim.Adam(modelo.parameters(), lr=0.001)
```

Adam: É o "professor" que atualiza os pesos da rede. `lr é a velocidade de aprendizado`. **O objetivo principal do Otimizador é fazer esse número do Loss chegar o mais perto possível de zero**.

```python
otimizador.zero_grad()
```

**O PyTorch, por padrão, acumula os erros das rodadas anteriores**. Se você não zerar a "memória" do otimizador a cada novo lote, a rede tentará corrigir erros que já foram tratados, o que faria o aprendizado explodir.

```python
mascaras_previstas = modelo(imagens)
```
**Aqui a imagem entra pela esquerda da U-Net, passa pelo "U" e sai pela direita como uma tentativa de máscara**. No início, esse chute é aleatório e horrível.

```python
erro = criterio(mascaras_previstas, mascaras_reais)
```

**Calcula a diferença entre o chute da U-Net e o gabarito real**

```python
erro.backward()
```

**Calcula como cada peso matemático deve ser ajustado (Backpropagation)**. O algoritmo viaja da saída para a entrada da rede, calculando exatamente quanto cada "neurônio" contribuiu para o erro final.

```python
otimizador.step()
```

Após descobrir quem errou (no passo anterior), **o Adam atualiza os pesos de cada neurônio**. Os que ajudaram a acertar são fortalecidos; os que causaram o erro são enfraquecidos.

```python
torch.save(modelo.state_dict(), caminho_salvar)
```

Em vez de salvar a rede inteira (que incluiria estruturas de código complexas e pesadas), **o `state_dict` salva apenas os números puros. Isso torna o arquivo .pth muito leve** (cerca de 14MB a 30MB para MobileNetV2) e fácil de carregar em qualquer outro computador, mesmo que ele não tenha acesso à internet ou ao seu script original do Colab.