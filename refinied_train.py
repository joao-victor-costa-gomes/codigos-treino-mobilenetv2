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
# CRIAÇÃO DA CLASSE DOS DATASETS (AGORA COM AUGMENTATION)
# ---------------------------------------------------------------- 

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image

class DatasetFire(Dataset):
    # Adicionamos a flag is_train para saber se aplicamos a mutação ou não
    def __init__(self, split_dir, is_train=False):
        self.split_dir = split_dir
        self.is_train = is_train
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

        # 1. Redimensionamento Padrão
        image = TF.resize(image, (224, 224))
        mask = TF.resize(mask, (224, 224), interpolation=Image.NEAREST)

        # 2. DATA AUGMENTATION (Aplicado apenas se for a pasta de Treino)
        if self.is_train:
            # 50% de chance de espelhar horizontalmente
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # 50% de chance de espelhar verticalmente (fogo de cabeça para baixo!)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

        # 3. Conversão para Tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask = torch.where(mask > 0, 1.0, 0.0)

        return image, mask

# ----------------------------------------------------------------
# DECLARANDO OS DATASETS
# ---------------------------------------------------------------- 

DIR_TRAIN = 'fire_dataset/train/'
# Avisamos que este é o treino, então ele pode mutar as imagens
dataset_train = DatasetFire(DIR_TRAIN, is_train=True)

DIR_VAL = 'fire_dataset/valid/'
# Avisamos que é validação, fotos devem ficar estáticas
dataset_val = DatasetFire(DIR_VAL, is_train=False)

dataloader_train = DataLoader(dataset_train, batch_size=4, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False)

print(f"Total de PARES para TREINO: {len(dataset_train)}")
print(f"Total de PARES para VALIDAÇÃO: {len(dataset_val)}")

# ----------------------------------------------------------------
# LOOP DE TREINAMENTO (AGORA COM SCHEDULER)
# ---------------------------------------------------------------- 

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"O modelo vai treinar usando: {device}")

modelo = modelo.to(device)

criterio = nn.BCEWithLogitsLoss()
otimizador = optim.Adam(modelo.parameters(), lr=0.001)

# O Freio ABS: Se o erro de validação não cair por 2 épocas, reduz a velocidade (LR) por 10
scheduler = optim.lr_scheduler.ReduceLROnPlateau(otimizador, mode='min', factor=0.1, patience=2)

# ONDE COMEÇA O LOOP
epocas = 50
paciencia_maxima = 6 # Aumentei levemente a paciência para dar tempo do Freio agir
epocas_sem_melhorar = 0
melhor_loss = float('inf')
caminho_salvar = 'modelos/semantic_fire_model_V2.pth' # Nome novo!

for epoca in range(epocas):
    # ===============================
    # FASE 1: TREINAMENTO
    # ===============================
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

    # ===============================
    # FASE 2: VALIDAÇÃO
    # ===============================
    modelo.eval()
    erro_val_total = 0.0

    with torch.no_grad():
        for imagens_val, mascaras_reais_val in dataloader_val:
            imagens_val = imagens_val.to(device)
            mascaras_reais_val = mascaras_reais_val.to(device)
            
            mascaras_previstas_val = modelo(imagens_val)
            erro_val = criterio(mascaras_previstas_val, mascaras_reais_val)
            erro_val_total += erro_val.item()
            
    erro_val_medio = erro_val_total / len(dataloader_val)

    # ===============================
    # FASE 3: AVALIAÇÃO E AJUSTES
    # ===============================
    # Pega a velocidade atual para mostrar no print
    lr_atual = otimizador.param_groups[0]['lr']
    print(f"Época {epoca+1}/{epocas} [Velocidade: {lr_atual:.6f}] - Erro Treino: {erro_treino_medio:.4f} | Erro Validação: {erro_val_medio:.4f}")

    # Aciona o Freio (Scheduler) baseado na nota da prova
    scheduler.step(erro_val_medio)

    if erro_val_medio < melhor_loss:
        print(f"  🌟 Novo recorde na PROVA! Erro caiu para {erro_val_medio:.4f}. Salvando V2...")
        melhor_loss = erro_val_medio
        epocas_sem_melhorar = 0 
        torch.save(modelo.state_dict(), caminho_salvar)
        
    else:
        epocas_sem_melhorar += 1
        print(f"  ⚠️ Não melhorou. Paciência: {epocas_sem_melhorar}/{paciencia_maxima}")
        if epocas_sem_melhorar >= paciencia_maxima:
            print("\n🚨 EARLY STOPPING ATIVADO! O modelo deu o seu melhor.")
            break