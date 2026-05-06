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
# CRIAÇÃO DA CLASSE DOS DATASETS
# ---------------------------------------------------------------- 

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image


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

# ----------------------------------------------------------------
# DECLARANDO OS DATASETS
# ---------------------------------------------------------------- 

DIR_TRAIN = 'fire_dataset/train/'
dataset_train = Dataset(DIR_TRAIN)

DIR_VAL = 'fire_dataset/valid/'
dataset_val = Dataset(DIR_VAL)

dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False)

print(f"Total de PARES para TREINO: {len(dataset_train)}")
print(f"Total de PARES para VALIDAÇÃO: {len(dataset_val)}")

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
caminho_salvar = 'modelos/semantic_fire_model_V1.pth'

for epoca in range(epocas):
    # FASE 1: TREINAMENTO (Estudar)
    modelo.train() 
    erro_treino_total = 0.0

    for imagens, mascaras_reais in dataloader_train:
        imagens = imagens.to(device)
        mascaras_reais = mascaras_reais.to(device)

        otimizador.zero_grad()
        mascaras_previstas = modelo(imagens)
        erro = criterio(mascaras_previstas, mascaras_reais)
        erro.backward()
        otimizador.step()

        erro_treino_total += erro.item()

    erro_treino_medio = erro_treino_total / len(dataloader_train)

    # FASE 2: VALIDAÇÃO (A Prova)
    modelo.eval()
    erro_val_total = 0.0

    with torch.no_grad(): # desliga o cálculo de gradientes (economiza muita memória)
        for imagens_val, mascaras_reais_val in dataloader_val:
            imagens_val = imagens_val.to(device)
            mascaras_reais_val = mascaras_reais_val.to(device)
            
            mascaras_previstas_val = modelo(imagens_val)
            erro_val = criterio(mascaras_previstas_val, mascaras_reais_val)
            erro_val_total += erro_val.item()
            
    erro_val_medio = erro_val_total / len(dataloader_val)

    # FASE 3: AVALIAÇÃO E SALVAMENTO
    print(f"Época {epoca+1}/{epocas} - Erro Treino: {erro_treino_medio:.4f} | Erro Validação: {erro_val_medio:.4f}")

    # relação de paciência com validação para impedir que o modelo decore
    if erro_val_medio < melhor_loss:
        print(f"  🌟 Novo recorde na PROVA! Erro caiu de {melhor_loss:.4f} para {erro_val_medio:.4f}. Salvando...")
        melhor_loss = erro_val_medio
        epocas_sem_melhorar = 0 
        torch.save(modelo.state_dict(), caminho_salvar)
        
    else:
        epocas_sem_melhorar += 1
        print(f"  ⚠️ Piorou na prova (Sinal de decoreba). Paciência: {epocas_sem_melhorar}/{paciencia_maxima}")
        if epocas_sem_melhorar >= paciencia_maxima:
            print("\n🚨 EARLY STOPPING ATIVADO! O modelo começou a decorar e parou de generalizar.")
            break