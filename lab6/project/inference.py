# inference.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import timm
from pathlib import Path
from typing import Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import argparse
import sys

# ------------------------------
# 1. Конфигурация устройства
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ------------------------------
# 2. Трансформация и DINOv2 (ленивая загрузка)
# ------------------------------
_transform = None
_dino_model = None

def get_transform():
    global _transform
    if _transform is None:
        _transform = A.Compose([
            A.PadIfNeeded(min_height=518, min_width=518, border_mode=0,
                          fill=(0, 0, 0), position="center"),
            A.Resize(518, 518),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    return _transform

def get_dino_model():
    global _dino_model
    if _dino_model is None:
        print("Loading DINOv2 model...")
        _dino_model = timm.create_model(
            "vit_base_patch14_dinov2", 
            pretrained=True,
            num_classes=0, 
            dynamic_img_size=True
        )
        _dino_model.eval().to(DEVICE)
        print("DINOv2 loaded.")
    return _dino_model

# ------------------------------
# 3. Архитектура классификатора
# ------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TCNTransformer(nn.Module):
    def __init__(self, input_dim=768, hidden=384, num_classes=3, dropout=0.4,
                 num_layers=3, kernel_size=3):
        super().__init__()
        tcn_blocks = []
        in_channels = input_dim
        for i in range(num_layers):
            dilation = 2 ** i
            tcn_blocks.append(TemporalBlock(in_channels, hidden, kernel_size,
                                            dilation, dropout))
            in_channels = hidden
        self.tcn = nn.Sequential(*tcn_blocks)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_enc = PositionalEncoding(hidden, max_len=9)
        nhead = 8 if hidden % 8 == 0 else 4
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes)
        )

    def forward(self, x):
        # x: [B, 8, 768]
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        cls_out = x[:, 0, :]
        return self.head(cls_out)

# ------------------------------
# 4. Загрузка ансамбля из 5 фолдов
# ------------------------------
def load_ensemble(models_dir: str, device: torch.device = DEVICE):
    """Загружает все 5 моделей из папки."""
    models = []
    for fold in range(5):
        ckpt_path = Path(models_dir) / f"fold_{fold}_best.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        # Важно: загружаем на CPU, потом переносим на нужное устройство
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        cfg = checkpoint.get('config', {})
        
        model = TCNTransformer(
            input_dim=768,
            hidden=cfg.get('hidden', 448),
            num_classes=3,
            dropout=cfg.get('dropout', 0.3),
            num_layers=cfg.get('tcn_layers', 2),
            kernel_size=cfg.get('kernel_size', 3)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval().to(device)
        models.append(model)
        print(f"Loaded fold {fold} (best F1 = {checkpoint.get('best_f1', 'N/A'):.4f})")
    return models

# ------------------------------
# 5. Извлечение эмбеддингов
# ------------------------------
@torch.no_grad()
def extract_embedding(image_path: str) -> np.ndarray:
    """DINOv2 эмбеддинг одного изображения."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = get_transform()
    x = transform(image=image)["image"]
    x = x.unsqueeze(0).to(DEVICE)
    
    dino = get_dino_model()
    emb = dino(x)
    return emb.squeeze(0).cpu().numpy().astype(np.float32)

def folder_to_sequence(folder_path: str) -> np.ndarray:
    """Берёт первые 8 картинок из папки → [8, 768]."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    image_files = sorted([
        f for f in folder.iterdir() 
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ])[:8]

    if len(image_files) != 8:
        raise ValueError(f"Expected 8 images in {folder_path}, found {len(image_files)}")

    print(f"Extracting embeddings from {len(image_files)} images...")
    embeddings = []
    for img_path in image_files:
        emb = extract_embedding(img_path)
        embeddings.append(emb)

    seq = np.stack(embeddings, axis=0)  # [8, 768]
    return seq

# ------------------------------
# 6. Ансамблевый предикт
# ------------------------------
class ActionEnsembleInference:
    def __init__(self, models_dir: str = "./models", device: torch.device = DEVICE):
        self.device = device
        self.models = load_ensemble(models_dir, device)
        self.class_names = {0: "inaction", 1: "move", 2: "work"}

    def predict_folder(self, folder_path: str) -> Dict:
        """Принимает путь к папке с 8 картинками, возвращает предсказание."""
        print(f"\n{'='*50}")
        print(f"Processing folder: {folder_path}")
        print(f"{'='*50}")
        
        # Извлекаем эмбеддинги
        seq = folder_to_sequence(folder_path)
        x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Ансамбль
        all_probs = []
        with torch.no_grad():
            for i, model in enumerate(self.models):
                logits = model(x)
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                pred = torch.argmax(logits, dim=-1).item()
                print(f"  Fold {i}: {self.class_names[pred]:10s} (prob={probs[0, pred].item():.4f})")

        # Усредняем вероятности
        avg_probs = np.mean(all_probs, axis=0).squeeze(0)
        final_class = int(np.argmax(avg_probs))
        final_confidence = avg_probs[final_class]

        print(f"\n  {'─'*40}")
        print(f"  FINAL: {self.class_names[final_class].upper()}")
        print(f"  Confidence: {final_confidence:.4f}")
        print(f"  Probabilities: inaction={avg_probs[0]:.4f}, move={avg_probs[1]:.4f}, work={avg_probs[2]:.4f}")
        print(f"  {'─'*40}")

        return {
            "class": self.class_names[final_class],
            "class_id": final_class,
            "confidence": float(final_confidence),
            "probabilities": {
                "inaction": float(avg_probs[0]),
                "move": float(avg_probs[1]),
                "work": float(avg_probs[2])
            },
            "fold_votes": [int(np.argmax(p)) for p in all_probs]
        }

# ------------------------------
# 7. CLI
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Action classifier from 8 frames")
    parser.add_argument("--images", type=str, required=True,
                        help="Path to folder with 8 images")
    parser.add_argument("--models", type=str, default="./models",
                        help="Path to folder with 5 fold checkpoints")
    args = parser.parse_args()

    # Проверяем, что папка с изображениями существует
    if not Path(args.images).exists():
        print(f"Error: folder not found: {args.images}")
        sys.exit(1)

    # Проверяем, что папка с моделями существует
    if not Path(args.models).exists():
        print(f"Error: models folder not found: {args.models}")
        sys.exit(1)

    engine = ActionEnsembleInference(models_dir=args.models)
    result = engine.predict_folder(args.images)
    
    print(f"\nResult: {result['class']} (confidence={result['confidence']:.4f})")