import os
import json
import docx
from pathlib import Path
from typing import List

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
)
from transformers.modeling_outputs import BaseModelOutput

# Import des métriques
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer
import bert_score
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize

# Téléchargement des ressources NLTK nécessaires (à faire une seule fois)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# ==========================
# Config
# ==========================
JSON_PATH = r"E:\MammoLLM\test_set_stratified.json" # Changez ceci vers votre fichier de test
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T5_MODEL_NAME = "t5-base"
BERT_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
IMAGE_SIZE = 224
SEQ_LEN_ENCODER = 16
MAX_REPORT_TOKENS = 256
SAVE_DIR = "./save_saved_multimodal_t5" # Assurez-vous que c'est le même dossier que l'entrainement
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================
# Utilitaires
# ==========================

def read_docx_text(path: str) -> str:
    try:
        doc = docx.Document(path)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)
    except Exception as e:
        print(f"Erreur lecture docx {path}: {e}")
        return ""

# ==========================
# Dataset
# ==========================

image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

class MammoReportDataset(Dataset):
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
        patient_id = e.get('patient_id', 'INCONNU')

        try:
            pil = Image.open(img_path).convert('RGB')
        except Exception as exc:
            pil = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
            print(f"Warning: could not open image {img_path}: {exc}")

        if self.transform:
            img = self.transform(pil)
        else:
            img = transforms.ToTensor()(pil)

        report_text = ""
        if report_path and Path(report_path).exists():
            report_text = read_docx_text(report_path)
        else:
            report_text = e.get('report', "")

        # Tokenisation pour le calcul de la loss (et pour la vérité terrain dans le DataLoader)
        tokenized = self.t5_tokenizer(report_text,
                                     truncation=True,
                                     padding='max_length',
                                     max_length=MAX_REPORT_TOKENS,
                                     return_tensors='pt')
        labels = tokenized.input_ids.squeeze(0)
        # T5 utilise -100 pour ignorer le calcul de la loss sur les tokens de padding
        labels[labels == self.t5_tokenizer.pad_token_id] = -100
        attention_mask_labels = tokenized.attention_mask.squeeze(0)

        return {
            'image': img,
            'complaint': complaint,
            'report_text': report_text,
            'labels': labels,
            'labels_mask': attention_mask_labels,
            'patient_id': patient_id,
        }

def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    complaints = [b['complaint'] for b in batch]
    report_texts = [b['report_text'] for b in batch]
    labels = torch.stack([b['labels'] for b in batch])
    labels_mask = torch.stack([b['labels_mask'] for b in batch])
    patient_ids = [b['patient_id'] for b in batch]
    return {'images': images, 'complaints': complaints, 'report_texts': report_texts, 'labels': labels, 'labels_mask': labels_mask, 'patient_ids': patient_ids}

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
        
        # Utilisez une couche de projection pour chaque modalité
        self.img_proj = nn.Linear(img_out_dim, t5_d_model)
        self.txt_proj = nn.Linear(bert_out_dim, t5_d_model)

        # Créez une couche de fusion plus significative (par ex. un transformateur)
        self.fusion_transform = nn.TransformerEncoderLayer(
            d_model=t5_d_model,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.fusion_transform, num_layers=2)

    def forward_backbone(self, images, complaints_inputs):
        img_feats = self.image_model(images)
        img_proj = self.img_proj(img_feats).unsqueeze(1)

        with torch.no_grad():
            bert_outputs = self.text_model(**complaints_inputs)
            txt_cls = bert_outputs.last_hidden_state[:, 0, :]
        txt_proj = self.txt_proj(txt_cls).unsqueeze(1)
        
        # Utilisez le transformateur pour fusionner les caractéristiques
        fused = torch.cat([img_proj, txt_proj], dim=1)
        fused_output = self.transformer_encoder(fused)
        
        return fused_output

    def forward(self, fused_vector):
        return fused_vector

# ==========================
# Helper: build image backbone (inchangé)
# ==========================
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        modules = list(rn.children())[:-1]
        self.backbone = nn.Sequential(*modules)

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.view(feat.size(0), -1)
        return feat

# ==========================
# Fonctions de calcul des métriques (inchangé)
# ==========================
def calculate_individual_bleu_scores(generated_reports, ground_truth_reports):
    references_for_bleu = [[text] for text in ground_truth_reports]
    
    bleu_1_metric = BLEU(max_ngram_order=1)
    bleu_1_score = bleu_1_metric.corpus_score(generated_reports, references_for_bleu)
    
    bleu_2_metric = BLEU(max_ngram_order=2)
    bleu_2_score = bleu_2_metric.corpus_score(generated_reports, references_for_bleu)
    
    bleu_3_metric = BLEU(max_ngram_order=3)
    bleu_3_score = bleu_3_metric.corpus_score(generated_reports, references_for_bleu)
    
    bleu_4_metric = BLEU(max_ngram_order=4)
    bleu_4_score = bleu_4_metric.corpus_score(generated_reports, references_for_bleu)
    
    return {
        'bleu_1': bleu_1_score.score / 100,
        'bleu_2': bleu_2_score.score / 100,
        'bleu_3': bleu_3_score.score / 100,
        'bleu_4': bleu_4_score.score / 100
    }

def calculate_meteor_score(generated_reports, ground_truth_reports):
    meteor_scores = []
    for gen_text, ref_text in zip(generated_reports, ground_truth_reports):
        try:
            gen_tokens = word_tokenize(gen_text.lower())
            ref_tokens = word_tokenize(ref_text.lower())
            score = meteor_score([ref_tokens], gen_tokens)
            meteor_scores.append(score)
        except Exception as e:
            print(f"Erreur lors du calcul METEOR: {e}")
            meteor_scores.append(0.0)
    
    return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0

# ==========================
# Evaluation (MIS À JOUR)
# ==========================
def evaluate(json_path: str, checkpoint_path: str):
    print(f"Utilisation de l'appareil: {DEVICE}")
    print("Démarrage de l'évaluation...")

    t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL_NAME)
    bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
   
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_MODEL_NAME).to(DEVICE)
    bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    img_backbone = ResNetFeatureExtractor(pretrained=True).to(DEVICE)
    multimodal = MultiModalToT5(image_model=img_backbone,
                                 text_model=bert_model,
                                 t5_d_model=t5_model.config.d_model,
                                 bert_out_dim=bert_model.config.hidden_size,
                                 img_out_dim=2048,
                                 seq_len=SEQ_LEN_ENCODER).to(DEVICE)

    if not os.path.exists(checkpoint_path):
        print(f"Erreur: Fichier de checkpoint non trouvé à {checkpoint_path}")
        return
    print(f"Chargement du checkpoint depuis {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    t5_model.load_state_dict(checkpoint['t5_state_dict'])
    multimodal.load_state_dict(checkpoint['multimodal_state_dict'])

    t5_model.eval()
    multimodal.eval()

    test_dataset = MammoReportDataset(json_path, tokenizer_t5=t5_tokenizer, transform=image_transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # ========================================================
    # --- CALCUL DE LA PERTE DE TEST (AJOUT) ---
    # ========================================================
    total_test_loss = 0.0
    
    print("\nCalcul de la perte de Test...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test Loss"):
            images = batch['images'].to(DEVICE)
            complaints = batch['complaints']
            labels = batch['labels'].to(DEVICE)
            
            # Tokenisation du complaint (texte)
            bert_inputs = bert_tokenizer(complaints, return_tensors='pt', padding=True, truncation=True, max_length=128)
            bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

            # 1. Obtenir les hidden states du multimodal encoder
            fused_output = multimodal.forward_backbone(images, bert_inputs)
            encoder_hidden_states = multimodal(fused_output)
            encoder_outputs = (encoder_hidden_states,)
            
            # 2. Calculer la perte avec T5 (labels sont requis pour la loss)
            outputs = t5_model(encoder_outputs=encoder_outputs, labels=labels)
            loss = outputs.loss
            
            # Multiplier par la taille du batch pour une moyenne pondérée
            total_test_loss += loss.item() * images.size(0) 

    avg_test_loss = total_test_loss / len(test_dataset)
    print(f"Perte de Test (Avg Loss) : {avg_test_loss:.4f}")

    # ========================================================
    # --- FIN DU CALCUL DE LA PERTE ---
    # ========================================================
    
    all_generated_reports = []
    all_ground_truth_reports = []
   
    print("\nGénération des rapports pour le jeu de test et affichage des résultats...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Génération des rapports"):
            images = batch['images'].to(DEVICE)
            complaints = batch['complaints']
            ground_truth_reports = batch['report_texts']
            patient_ids = batch['patient_ids']

            bert_inputs = bert_tokenizer(complaints, return_tensors='pt', padding=True, truncation=True, max_length=128)
            bert_inputs = {k: v.to(DEVICE) for k, v in bert_inputs.items()}

            fused_output = multimodal.forward_backbone(images, bert_inputs)
            encoder_hidden_states = multimodal(fused_output)
            
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_hidden_states,
                hidden_states=None,
                attentions=None
            )
           
            # Génération du rapport T5
            generated = t5_model.generate(
                encoder_outputs=encoder_outputs,
                max_length=MAX_REPORT_TOKENS,
                num_beams=4,
                early_stopping=True,
                pad_token_id=t5_tokenizer.pad_token_id,
                eos_token_id=t5_tokenizer.eos_token_id
            )

            decoded_reports = [t5_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated]

            for i in range(len(decoded_reports)):
                # Affichage des exemples
                # print(f"--- Échantillon {patient_ids[i]} ---")
                # print(f"Vérité terrain: {ground_truth_reports[i]}")
                # print(f"Rapport généré: {decoded_reports[i]}")
                # print("-" * 50)
                pass # Désactiver l'affichage des 50 lignes pour la version finale

           
            all_generated_reports.extend(decoded_reports)
            all_ground_truth_reports.extend(ground_truth_reports)

    print("\nCalcul des scores d'évaluation...")

    # ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for gen_text, ref_text in zip(all_generated_reports, all_ground_truth_reports):
        scores = scorer.score(ref_text, gen_text)
        rouge_scores['rouge1'] += scores['rouge1'].fmeasure
        rouge_scores['rouge2'] += scores['rouge2'].fmeasure
        rouge_scores['rougeL'] += scores['rougeL'].fmeasure
   
    num_samples = len(all_generated_reports)
    if num_samples > 0:
        rouge_scores = {k: v / num_samples for k, v in rouge_scores.items()}

    # BLEU scores individuels (1, 2, 3, 4)
    print("Calcul des scores BLEU...")
    bleu_scores = calculate_individual_bleu_scores(all_generated_reports, all_ground_truth_reports)

    # METEOR score
    print("Calcul du score METEOR...")
    meteor_avg_score = calculate_meteor_score(all_generated_reports, all_ground_truth_reports)

    # BERTScore
    print("Calcul du score BERTScore...")
    # NOTE: Assurez-vous que le modèle BERTScore supporte le français ('lang="fr"')
    P, R, F1 = bert_score.score(all_generated_reports, all_ground_truth_reports, lang="fr", verbose=True) 

    # Affichage des résultats (MIS À JOUR)
    print("\n" + "="*60)
    print("RÉSUMÉ DES SCORES D'ÉVALUATION")
    print("="*60)
    
    print("\n--- PERFORMANCE GLOBALE ---")
    print(f"Perte de Test (Loss)   : {avg_test_loss:.4f}") # AJOUT DE LA LOSS
    
    print("\n--- SCORES ROUGE (F-mesure) ---")
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")

    print("\n--- SIMILARITÉ LEXICALE ---")
    print(f"BLEU-1: {bleu_scores['bleu_1']:.4f}")
    print(f"BLEU-4: {bleu_scores['bleu_4']:.4f}")
    print(f"METEOR: {meteor_avg_score:.4f}")
    
    print("\n--- SIMILARITÉ SÉMANTIQUE ---")
    print(f"BERTScore F1: {F1.mean().item():.4f}")
    
    print("="*60)
    print("Évaluation terminée.")

if __name__ == '__main__':
    TEST_JSON_PATH = r"E:\MammoLLM\test_set_stratified.json" # Changez ceci si besoin
    # Assurez-vous que ce chemin pointe vers votre meilleur checkpoint entraîné
    CHECKPOINT_TO_EVAL = r"E:\MammoLLM\new\save_saved_multimodal_t5_finetuned\multimodal_t5_epoch10.pt" 
    
    evaluate(TEST_JSON_PATH, CHECKPOINT_TO_EVAL)