# Descrição dos parâmetros da classe 'Unet'

---

### 1. Os Parâmetros Essenciais (Você DEVE usar)

Estes são os parâmetros que ditam a estrutura básica do modelo e o que ele vai receber e prever.

`encoder_name (str)`
- **O que é:** É o motor de extração de características (o backbone).
- **Seu uso:** "mobilenet_v2". É vital para o seu PC sem placa de vídeo, pois garante uma rede leve.

`encoder_weights (str | None)`
**O que é:** Define se o motor vem pré-treinado ou "zerado".
**Seu uso:** `"imagenet"`. Como já discutimos, é o "fotógrafo adulto" que acelera seu treino exponencialmente.

`in_channels (int)`
**O que é:** Quantos canais de cor tem a imagem de entrada.
**Seu uso:** 3. Suas fotos de celular e de webcam sempre serão RGB (Vermelho, Verde, Azul).

`classes (int)`
**O que é:** Quantas "camadas" de máscara a rede vai cuspir no final.
**Seu uso:** `1`. Você quer apenas saber "Tem fogo?" (`1`) ou "Não tem?" (`0`).

---

### 2. Parâmetros de Ajuste (Você PODE usar para otimizar)

Se o seu protótipo estiver lento no PC da indústria, você pode usar esses parâmetros para criar uma U-Net menor ou mais rápida.

`encoder_depth (int)`
**O que é:** Controla quantos "degraus" de descida o lado esquerdo da U-Net vai ter. O padrão é 5 (a imagem encolhe 5 vezes).
**Como usar:** Se o modelo estiver exigindo muita RAM ou rodando a poucos FPS (quadros por segundo) na câmera, você pode mudar para encoder_depth=4. Isso remove a camada mais profunda (o gargalo mais pesado), tornando o modelo incrivelmente rápido (embora perca um pouco de noção de contexto global).

`decoder_channels (Sequence[int])`
**O que é:** Diz quantos canais (filtros) haverá no lado da subida (Decoder). O padrão para profundidade 5 é (256, 128, 64, 32, 16).
**Como usar:** Só mexa aqui se você alterar o encoder_depth. Por exemplo, se colocar a profundidade para 4, você precisará passar uma lista menor: decoder_channels=(128, 64, 32, 16).

`activation (str | None)`
**O que é:** Uma função matemática aplicada logo na saída da rede.
**Como usar:** O padrão é None (devolve números crus). Se você quiser que o modelo já devolva probabilidades (de 0% a 100%), você pode passar activation="sigmoid". Nota: Se você fizer isso, você deve tirar o torch.sigmoid() que escrevemos no código de visualização para não aplicar duas vezes.

---

### 3. Parâmetros Avançados (Você NÃO PRECISA usar agora)

Estas opções existem para pesquisadores ou cenários muito específicos. Deixe-as com os valores padrão para o seu protótipo.

`decoder_use_norm (bool | str | dict)`
**O que é:** Aplica "Normalização" (como o Batch Normalization) no Decoder para evitar que os números explodam e fiquem muito grandes.
**Por que ignorar:** O padrão (True -> batchnorm) já é perfeito. Mexer aqui só é necessário se você estiver programando hardwares exóticos que não suportam batchnorm.

`decoder_attention_type (str | None)`
**O que é:** Adiciona módulos de "Atenção" (como scse). Faz o Decoder focar ativamente em partes importantes da imagem.
**Por que ignorar:** É legal para imagens médicas complexas (ex: achar um tumor escondido). Para chamas de forno, a chama já é brilhante e óbvia. Adicionar isso só deixaria o modelo mais pesado e lento no seu i5.

`decoder_interpolation (str)`
**O que é:** A técnica matemática usada para "esticar" a imagem no Upsampling (ex: nearest, bilinear).
**Por que ignorar:** O padrão nearest é rápido e eficiente. Mudanças aqui raramente impactam o IoU de forma significativa, mas podem aumentar o custo de processamento.

`aux_params (dict | None)`
**O que é:** Permite que a rede faça duas coisas ao mesmo tempo: prever a máscara (segmentação) E dizer o que é a imagem inteira (classificação).
**Por que ignorar:** Você não precisa classificar a imagem; você só quer a máscara. Deixe como None.

`kwargs`
**O que é:** Parâmetros extra passados apenas se você estivesse usando motores baseados em Transformers (uma tecnologia mais nova e pesada que CNNs).
**Por que ignorar:** Você está usando a MobileNetV2 (uma CNN clássica).

---