import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import segmentation_models_pytorch as smp


DIR_TEST = 'fire_dataset/test/'
CAMINHO_MODELO = 'modelos/semantic_fire_model_V1.pth'

# CLASSE DO DATASET 
class DatasetTest(Dataset):
    def __init__(self, split_dir):
        self.split_dir = split_dir
        self.images = [f for f in os.listdir(split_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace('.jpg', '_mask.png')
        img_path = os.path.join(self.split_dir, img_name)
        mask_path = os.path.join(self.split_dir, mask_name)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224), interpolation=Image.NEAREST)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)

        return image, mask

# FUNÇÃO MATEMÁTICA DO IOU
def calcular_iou(previsao, gabarito, epsilon=1e-6):
    # transforma probabilidades em 0 ou 1
    previsao = (previsao > 0.5).float()
    # achata os tensores para 1D (facilita a matemática)
    previsao = previsao.view(-1)
    gabarito = gabarito.view(-1)
    # Interseção: Onde a previsão e o gabarito são 1 ao mesmo tempo
    intersecao = (previsao * gabarito).sum()
    # União: Onde a previsão OU o gabarito são 1
    uniao = previsao.sum() + gabarito.sum() - intersecao
    # retorna o IoU (epsilon evita divisão por zero se a imagem for toda preta)
    return (intersecao + epsilon) / (uniao + epsilon)

# PREPARANDO O AMBIENTE E O MODELO
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Avaliando na: {device}")

dataset_test = DatasetTest(DIR_TEST)
dataloader_test = DataLoader(dataset_test, batch_size=4, shuffle=False)

print(f"Total de imagens para a Prova Final: {len(dataset_test)}\n")

modelo = smp.Unet(
    encoder_name="mobilenet_v2",        
    encoder_weights=None, # carregando modelo do zero
    in_channels=3,                      
    classes=1                           
)

modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=device))
modelo = modelo.to(device)
modelo.eval() # modo de avaliação

# AVALIAÇÃO DO MODELO
iou_total = 0.0
imagens_avaliadas = 0

print("Iniciando a avaliação cega. Por favor, aguarde...")

with torch.no_grad():
    for imagens, mascaras_reais in dataloader_test:
        imagens = imagens.to(device)
        mascaras_reais = mascaras_reais.to(device)
        # faz as previsões
        previsoes_brutas = modelo(imagens)
        previsoes_prob = torch.sigmoid(previsoes_brutas)
        # calcula o IoU de cada imagem do lote
        for i in range(len(imagens)):
            iou = calcular_iou(previsoes_prob[i], mascaras_reais[i])
            iou_total += iou.item()
            imagens_avaliadas += 1

iou_medio = iou_total / imagens_avaliadas
iou_porcentagem = iou_medio * 100

print("-" * 40)
print(" RESULTADO FINAL OFICIAL ")
print("-" * 40)
print(f"Imagens Testadas: {imagens_avaliadas}")
print(f"IoU Médio: {iou_medio:.4f} ({iou_porcentagem:.2f}%)")
print("-" * 40)

if iou_porcentagem > 70:
    print("Avaliação: EXCELENTE! 🌟 Modelo pronto para publicação/TCC.")
elif iou_porcentagem > 50:
    print("Avaliação: BOM! ✅ O modelo generalizou, mas pode melhorar.")
else:
    print("Avaliação: RUIM. ⚠️ O modelo ainda está confuso.")