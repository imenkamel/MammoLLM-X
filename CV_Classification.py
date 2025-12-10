import numpy as np
import pandas as pd
import os
import cv2
import json
import torch
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import AutoTokenizer, AutoModel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l2


import matplotlib.pyplot as plt

# ===============================================
# GPU DEVICE DETECTION
# ===============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device utilisé : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

# ===============================================
# CHARGEMENT DU DATASET JSON
# ===============================================
def load_mammography_dataset_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    image_paths, labels, complaints = [], [], []
    for entry in data:
        image_paths.append(entry["image"])
        labels.append(entry["label"])
        complaints.append(entry["complaints"])
    df = pd.DataFrame({"image_path": image_paths, "label": labels, "complaints": complaints})
    return df

# ===============================================
# PREPROCESSING IMAGES ET LABELS
# ===============================================
def normalize_grayscale(img):
    x_min, x_max = img.min(), img.max()
    return (img - x_min) / (x_max - x_min) if x_max > x_min else img

def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image non trouvée: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return normalize_grayscale(img)
    except Exception as e:
        print(f"Erreur {img_path}: {e}")
        return None

def load_images(df):
    images, labels, texts, skipped = [], [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(row['image_path'])
        if img is not None:
            images.append(img)
            labels.append(row['label'])
            texts.append(row['complaints'])
        else:
            skipped.append(row['image_path'])
    print(f"Images ignorées: {len(skipped)}")
    return np.array(images), np.array(labels), texts

def standardize_labels(labels):
    label_map = {"normal": "Normal", "benign": "Benign", "malignant": "Malignant",
                 "0": "Normal", "1": "Benign", "2": "Malignant"}
    return [label_map.get(str(lbl).lower(), lbl) for lbl in labels]

# ===============================================
# AUGMENTATION DE DONNEES
# ===============================================
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment_minority_classes(images, labels, target_samples=1000):
    labels = np.array(labels)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

    unique, counts = np.unique(labels_encoded, return_counts=True)
    max_count = max(counts)
    target_per_class = min(target_samples, max_count)

    augmented_images, augmented_labels = [], []

    for class_idx in unique:
        class_images = images[labels_encoded == class_idx]
        class_label = le.inverse_transform([class_idx])[0]
        to_generate = target_per_class - len(class_images)
        gen_count = 0

        while gen_count < to_generate:
            for img in class_images:
                if gen_count >= to_generate:
                    break
                for batch in datagen.flow(img.reshape((1,) + img.shape), batch_size=1):
                    aug_img = batch[0]
                    # Vérifie que l'image a bien 3 dimensions (224, 224, 3)
                    if isinstance(aug_img, np.ndarray) and aug_img.ndim == 3:
                        augmented_images.append(aug_img)
                        augmented_labels.append(class_label)
                        gen_count += 1
                    break  # important pour sortir de datagen.flow()

    # Vérifie les dimensions et concatène correctement
    if len(augmented_images) > 0:
        augmented_images = [img for img in augmented_images if isinstance(img, np.ndarray) and img.ndim == 3]
        images = np.concatenate([images, np.stack(augmented_images)], axis=0)
        labels = np.concatenate([labels, np.array(augmented_labels)], axis=0)

    return images, labels


# ===============================================
# MODELES
# ===============================================
def build_feature_extractor():
    base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    return Model(inputs=base.input, outputs=x)

def extract_features(images, model, batch_size=16):
    return model.predict(images, batch_size=batch_size, verbose=1)

def extract_text_features(texts, tokenizer, model, device, max_length=128, batch_size=16):
    model.eval()
    features = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extraction texte"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

            # Envoie chaque tenseur du batch sur le device
            inputs = {key: val.to(device) for key, val in inputs.items()}

            outputs = model(**inputs)
            # Utilise le CLS token ([0,0,:]) comme embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            features.append(cls_embeddings.cpu().numpy())

    return np.vstack(features)

def build_classifier(input_dim, num_classes, dropout_rate=0.3, l2_lambda=0.001):
    inputs = Input(shape=(input_dim,))

    # Couche cachée 1 avec L2 + Dropout
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_lambda))(inputs)
    x = Dropout(dropout_rate)(x)

    # Couche cachée 2 avec L2 + Dropout
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout_rate)(x)

    # Couche de sortie
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
# ===============================================
# EVALUATION
# ===============================================
def evaluate_model(model, X_test, y_test, le):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    cm = confusion_matrix(y_test, y_pred)
    acc = np.mean(y_pred == y_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"AUC: {auc:.3f} | Accuracy: {acc:.3f}")
    print("Matrice de confusion:\n", cm)

# ===============================================
# MAIN
# ===============================================
def main():
    json_path = r"E:\MammoLLM\CDD-CESM\newaug.json"
    df = load_mammography_dataset_from_json(json_path)
    print(f"Total images: {len(df)}")

    # Chargement des images
    images, labels, _ = load_images(df)
    labels = standardize_labels(labels)

    # Gestion des plaintes
    complaints = df["complaints"].tolist()

    # Augmentation des données image
    images, labels = augment_minority_classes(images, labels)

    # Compléter les plaintes pour les images augmentées
    original_count = len(df)
    augmented_count = len(images) - original_count
    complaints += complaints[:augmented_count]  # duplication naïve
    assert len(complaints) == len(images), "Erreur : plaintes et images non alignées"

    # Label encoding
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # Détection GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device utilisé : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Extraction des features image
    print("Extraction des features images...")
    fe_model = build_feature_extractor()
    features_img = extract_features(images, fe_model)

    # Extraction des features texte avec tqdm
    print("Extraction des features texte (Bio_ClinicalBERT)...")
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model_txt = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(device)
    features_txt = extract_text_features(complaints, tokenizer, model_txt, device)

    # Split train/test
    X_train_img, X_test_img, X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        features_img, features_txt, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fusion image + texte
    X_train = np.concatenate([X_train_img, X_train_txt], axis=1)
    X_test = np.concatenate([X_test_img, X_test_txt], axis=1)

    # Categorical targets
    from tensorflow.keras.utils import to_categorical
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    # Classifier MLP
    clf = build_classifier(X_train.shape[1], len(le.classes_))
    history = clf.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=500, batch_size=32)

    # Résultats
   


    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")
    plt.show()

    evaluate_model(clf, X_test, y_test, le)

if __name__ == "__main__":
    main()
