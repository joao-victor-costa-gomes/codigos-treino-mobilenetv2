# ----------------------------------------------------------------
# IMPLEMENTAÇÃO DA U-NET + BACKBONE MOBILENETV2 + PESOS (IMAGENET)
# ----------------------------------------------------------------

import segmentation_models_pytorch as smp

modelo = smp.Unet(
    encoder_name="mobilenet_v2",        
    encoder_weights="imagenet",         
    in_channels=3,                      
    classes=1                           
)

# ----------------------------------------------------------------
# PREPARAÇÃO DA CLASSE DO DATASET
# ---------------------------------------------------------------- 

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

DIR_TREINO = '...'

class Dataset(Dataset):
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

dataset = Dataset(DIR_TREINO)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Total de PARES (imagem/máscara) encontrados: {len(dataset)}")

# ----------------------------------------------------------------
# LOOP DE TREINAMENTO
# ---------------------------------------------------------------- 

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"O modelo vai treinar usando: {device}")

modelo = modelo.to(device)

criterio = nn.BCEWithLogitsLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

# ONDE COMEÇA O LOOP
epocas = 50
paciencia_maxima = 5
epocas_sem_melhorar = 0
melhor_loss = float('inf')
caminho_salvar = '...'

for epoca in range(epocas):
    modelo.train() # COLOCA REDE NO MODO APRENDIZADO
    erro_total = 0.0

    # PEGA LOTE DE 4 IMAGENS POR VEZ DO DATALOADER
    for imagens, mascaras_reais in dataloader:
        # MOVE OS DADOS PARA CPU/GPU
        # ONDE ESTÁ O MODELO AGORA
        imagens = imagens.to(device)
        mascaras_reais = mascaras_reais.to(device)

        otimizador.zero_grad()
        mascaras_previstas = modelo(imagens)
        erro = criterio(mascaras_previstas, mascaras_reais)
        erro.backward()
        otimizador.step()

        erro_total += erro.item()

    erro_medio = erro_total / len(dataloader)
    print(f"Época {epoca+1}/{epocas} - Erro Médio (Loss): {erro_medio:.4f}")

    if erro_medio < melhor_loss:
        print(f"  🌟 Novo recorde! O erro caiu de {melhor_loss:.4f} para {erro_medio:.4f}. Salvando o cérebro...")
        melhor_loss = erro_medio
        epocas_sem_melhorar = 0 
        torch.save(modelo.state_dict(), caminho_salvar)
        
    else:
        epocas_sem_melhorar += 1
        print(f"  ⚠️ O modelo não melhorou. Paciência: {epocas_sem_melhorar}/{paciencia_maxima}")
        if epocas_sem_melhorar >= paciencia_maxima:
            print("\n🚨 EARLY STOPPING ATIVADO! O modelo convergiu e parou de aprender.")
            print(f"O treinamento foi interrompido para economizar tempo. O seu melhor modelo seguro já está salvo no Drive.")
            break 

print("Treinamento finalizado!")

print(f"Sucesso! O cérebro da sua U-Net foi salvo fisicamente em: {caminho_salvar}")