import os
import json
import docx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from torch.optim import AdamW

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import BaseModelOutput

# ==========================
# Config
# ==========================
JSON_PATH = r"E:\MammoLLM\train_set_stratified.json"
BATCH_SIZE = 8
EPOCHS = 10
LR = 3e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T5_MODEL_NAME = "t5-base"
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE = 224
SEQ_LEN_ENCODER = 16
MAX_REPORT_TOKENS = 256
SAVE_DIR = "./save_saved_multimodal_t5_finetuned_loss_curve"
os.makedirs(SAVE_DIR, exist_ok=True)


# ==========================
# Fonctions d'Analyse des Paramètres (MISES À JOUR)
# ==========================
def count_module_params(module: nn.Module) -> int:
    """Compte les paramètres entraînables dans un module donné."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def detail_fusion_params(multimodal_model: nn.Module) -> int:
    """
    Calcule dynamiquement et affiche le détail du calcul des paramètres 
    pour les couches de Fusion/Projection (Catégorie 4).
    """
    
    # 1. Projections (Comptage dynamique)
    img_proj_count = count_module_params(multimodal_model.img_proj)
    txt_proj_count = count_module_params(multimodal_model.txt_proj)
    
    # 2. Transformer Encoder Layer (Comptage dynamique)
    fusion_layer = multimodal_model.fusion_transform
    fusion_layer_count = count_module_params(fusion_layer)
    num_layers = multimodal_model.transformer_encoder.num_layers
    
    # Le total est la somme des parties comptées dynamiquement
    total_fusion_count = img_proj_count + txt_proj_count + (fusion_layer_count * num_layers)

    # Récupération des dimensions pour l'affichage pédagogique
    t5_d_model = multimodal_model.t5_d_model
    img_out_dim = multimodal_model.img_proj.in_features
    bert_out_dim = multimodal_model.txt_proj.in_features

    print("\n" + "=" * 70)
    print("Détail du Calcul des Paramètres de Fusion/Projection (Décompte Dynamique) :")
    print("-" * 70)
    print(f"Dimensions utilisées (d_model T5): {t5_d_model}")
    print(f"Dim. Image (ResNet): {img_out_dim}, Dim. Texte (BERT): {bert_out_dim}")
    print("-" * 70)
    
    print(f"1. Projection Image (`img_proj`): {img_proj_count:,}")
    print(f"   (Théorie: ({img_out_dim} * {t5_d_model}) + {t5_d_model} Bias = 1,573,632)")
    
    print(f"2. Projection Texte (`txt_proj`): {txt_proj_count:,}")
    print(f"   (Théorie: ({bert_out_dim} * {t5_d_model}) + {t5_d_model} Bias = 590,592)")
    
    print(f"3. Transformer Encoder Fusion ({num_layers} couches): {fusion_layer_count * num_layers:,}")
    print(f"   ({fusion_layer_count:,} params par couche. Théorie totale = 11,027,968)")
    
    print("-" * 70)
    print(f"TOTAL DYNAMIQUE DE LA FUSION (Catégorie 4) : {total_fusion_count:,}")
    print("=" * 70)
    
    return total_fusion_count


# ==========================
# Utilitaires (inchangé)
# ==========================
def read_docx_text(path: str) -> str:
    """Read all text from a docx file."""
    try:
        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        # print(f"Erreur lecture docx {path}: {e}")
        return ""

# ==========================
# Dataset (inchangé)
# ==========================
image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class MammoReportDataset(Dataset):
    """Custom dataset for loading images, complaints, and medical reports."""
    def __init__(self, json_path: str, tokenizer_t5: T5Tokenizer, transform=None):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.transform = transform
        self.t5_tokenizer = tokenizer_t5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = self.data[idx]
        img_path = e['image']
        complaint = e.get('complaints', "")
        report_path = e.get('medical_report_path', None)

        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception as exc:
            pil = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            # print(f"Warning: could not open image {img_path}: {exc}")

        if self.transform:
            img = self.transform(pil)
        else:
            img = transforms.ToTensor()(pil)

        report_text = ""
        if report_path and Path(report_path).exists():
            report_text = read_docx_text(report_path)
        else:
            report_text = e.get('report', "")

        tokenized = self.t5_tokenizer(report_text,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=MAX_REPORT_TOKENS,
                                          return_tensors='pt')
        labels = tokenized.input_ids.squeeze(0)
        labels[labels == self.t5_tokenizer.pad_token_id] = -100
        attention_mask_labels = tokenized.attention_mask.squeeze(0)

        return {
            'image': img,
            'complaint': complaint,
            'report_text': report_text,
            'labels': labels,
            'labels_mask': attention_mask_labels,
        }

def collate_fn(batch):
    """Custom collate function to handle batching."""
    images = torch.stack([b['image'] for b in batch])
    complaints = [b['complaint'] for b in batch]
    labels = torch.stack([b['labels'] for b in batch])
    labels_mask = torch.stack([b['labels_mask'] for b in batch])
    return {'images': images, 'complaints': complaints, 'labels': labels, 'labels_mask': labels_mask}

# ==========================
# Multimodal model wrapper (inchangé)
# ==========================
class MultiModalToT5(nn.Module):
    def __init__(self,
                 image_model: nn.Module,
                 text_model: nn.Module,
                 t5_d_model: int = 768,
                 bert_out_dim: int = 768,
                 img_out_dim: int = 2048,
                 seq_len: int = SEQ_LEN_ENCODER):
        super().__init__()
        self.image_model = image_model
        self.text_model = text_model
        self.seq_len = seq_len
        self.t5_d_model = t5_d_model
        
        # Projection pour chaque modalité
        self.img_proj = nn.Linear(img_out_dim, t5_d_model)
        self.txt_proj = nn.Linear(bert_out_dim, t5_d_model)

        # Fusion via Transformer Encoder
        self.fusion_transform = nn.TransformerEncoderLayer(
            d_model=t5_d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True # Ajouté pour la cohérence PyTorch, si ce n'était pas là
        )
        self.transformer_encoder = nn.TransformerEncoder(self.fusion_transform, num_layers=2)

    def forward_backbone(self, images, complaints_inputs):
        img_feats = self.image_model(images)
        img_proj = self.img_proj(img_feats).unsqueeze(1)

        
        bert_outputs = self.text_model(**complaints_inputs)
        txt_cls = bert_outputs.last_hidden_state[:, 0, :]
        txt_proj = self.txt_proj(txt_cls).unsqueeze(1)
        
        # Fusion via Transformer Encoder
        fused = torch.cat([img_proj, txt_proj], dim=1)
        fused_output = self.transformer_encoder(fused)
        
        return fused_output

    def forward(self, fused_vector):
        return fused_vector

# ==========================
# Helper: build image backbone (inchangé)
# ==========================
class ResNetFeatureExtractor(nn.Module):
    """ResNet backbone to extract features without classifier."""
    def __init__(self, pretrained=True):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        # Supprime la couche de classification (la dernière)
        modules = list(rn.children())[:-1] 
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        return feat

# ==========================
# Training loop (MIS À JOUR)
# ==========================
def train():
    print(f"Using device: {DEVICE}")

    # 1. Initialisation des modèles
    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
    
    # --- GEL DE L'ENCODEUR T5 ---
    for param in t5_model.get_encoder().parameters():
        param.requires_grad = False
    print("INFO: L'encodeur T5 est gelé et ne sera pas entraîné.")
    # ----------------------------
    
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    
    # --- BERT RESTE ENTRAÎNABLE (par défaut, pas de gel) ---
    print("INFO: Le modèle BERT (Bio_ClinicalBERT) est ENTIÈREMENT entraînable (par défaut).")
    # -----------------------------------------------------

    img_backbone = ResNetFeatureExtractor(pretrained=True).to(DEVICE)

    multimodal = MultiModalToT5(
        image_model=img_backbone,
        text_model=bert_model,
        t5_d_model=t5_model.config.d_model,
        bert_out_dim=bert_model.config.hidden_size,
        img_out_dim=2048,
        seq_len=SEQ_LEN_ENCODER
    )
    multimodal.to(DEVICE)

    # 2. CALCUL DÉTAILLÉ DES PARAMÈTRES ENTRAÎNABLES (Utilise le comptage dynamique)
    
    # Paramètres de T5 (Décodeur seul)
    t5_params = count_module_params(t5_model.get_decoder()) 

    # Paramètres de BERT (Entièrement entraînable)
    bert_params = count_module_params(bert_model)

    # Paramètres de ResNet50 (Entièrement entraînable)
    resnet_params = count_module_params(img_backbone)

    # Paramètres des couches de fusion/projection (Comptage dynamique et détaillé)
    fusion_params = detail_fusion_params(multimodal)

    total_params = t5_params + bert_params + fusion_params + resnet_params
    
    # 3. Affichage du détail
    print("\n" + "=" * 60)
    print("Détail des paramètres entraînables du Modèle Multimodal :")
    print("-" * 60)
    print(f"  1. T5ForConditionalGeneration (Décodeur seul) : {t5_params:,}")
    print(f"  2. BERT (Bio_ClinicalBERT)                    : {bert_params:,}")
    print(f"  3. ResNet50 Backbone (Extracteur Image)       : {resnet_params:,}")
    print(f"  4. Couches de Fusion (Projections/Encoder)    : {fusion_params:,}")
    print("-" * 60)
    print(f"TOTAL GLOBAL DES PARAMÈTRES ENTRAÎNABLES        : {total_params:,}")
    print("=" * 60 + "\n")
    
    # 4. Préparation du DataLoaders
    dataset = MammoReportDataset(JSON_PATH, tokenizer_t5=t5_tokenizer, transform=image_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 5. Optimiseur et Scheduler
    
    # On rassemble tous les paramètres entraînables des différents modèles/modules
    params = list(multimodal.parameters()) + list(t5_model.parameters()) + list(bert_model.parameters()) + list(img_backbone.parameters())
    trainable_params = [p for p in params if p.requires_grad]
    
    optimizer = AdamW(trainable_params, lr=LR)

    total_steps = len(dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps
    )

    t5_model.train()
    multimodal.train()
    bert_model.train() 

    # --- INITIALISATION POUR LA COURBE DE PERTE ---
    epoch_losses = [] 
    # ----------------------------------------------

    # 6. Boucle d'entraînement
    global_step = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in pbar:
            images = batch['images'].to(DEVICE)
            complaints = batch['complaints']

            bert_inputs = bert_tokenizer(
                complaints, return_tensors='pt', padding=True,
                truncation=True, max_length=128
            )
            bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

            fused_output = multimodal.forward_backbone(images, bert_inputs)
            encoder_hidden_states = multimodal(fused_output)

            labels = batch['labels'].to(DEVICE)
            encoder_outputs = (encoder_hidden_states,)

            outputs = t5_model(encoder_outputs=encoder_outputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        
        # --- COLLECTE DE LA PERTE ---
        epoch_losses.append(avg_loss)
        # ----------------------------

        # 7. Sauvegarde du checkpoint
        ckpt_path = os.path.join(SAVE_DIR, f"multimodal_t5_epoch{epoch+1}.pt")
        torch.save({
            't5_state_dict': t5_model.state_dict(),
            'multimodal_state_dict': multimodal.state_dict(),
            'bert_state_dict': bert_model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch+1
        }, ckpt_path)
        print(f"Saved checkpoint {ckpt_path}")

    print("Training finished.")

    # ========================================================
    # --- AFFICHAGE GRAPHIQUE DE LA COURBE DE PERTE ---
    # ========================================================
    if epoch_losses:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, EPOCHS + 1), epoch_losses, marker='o', linestyle='-', color='blue')
        
        plt.title('Courbe de Perte (Training Loss) par Époque')
        plt.xlabel('Époque')
        plt.ylabel('Perte Moyenne (Loss)')
        plt.xticks(range(1, EPOCHS + 1)) 
        plt.grid(True)
        
        loss_curve_path = os.path.join(SAVE_DIR, "loss_curve.png")
        plt.savefig(loss_curve_path)
        print(f"\nCourbe de perte sauvegardée sous: {loss_curve_path}")

# ==========================
# Exécution du script
# ==========================
if __name__ == '__main__':
    train()