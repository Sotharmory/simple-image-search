import os
import numpy as np
import clip
import torch
from PIL import Image
import re

# Đường dẫn ảnh và nơi lưu feature
IMG_DIR = 'static/img'
FEATURE_PATH = 'static/clip_features.npy'
SUBJECT_PATH = 'static/clip_subjects.npy'

# Danh sách chủ đề (có thể mở rộng)
SUBJECTS = ['mèo', 'chó', 'sách', 'báo', 'hoa', 'cá', 'gấu', 'vịt', 'xe', 'rạp_phim', 'tranh', 'rừng', 'biển', 'IT', 'thời_trang', 'nhà']

def normalize_subject(s):
    return s.lower().replace('đ', 'd').replace('á', 'a').replace('à', 'a').replace('ả', 'a').replace('ã', 'a').replace('ạ', 'a').replace('â', 'a').replace('ă', 'a').replace('é', 'e').replace('è', 'e').replace('ẻ', 'e').replace('ẽ', 'e').replace('ẹ', 'e').replace('ê', 'e').replace('í', 'i').replace('ì', 'i').replace('ỉ', 'i').replace('ĩ', 'i').replace('ị', 'i').replace('ó', 'o').replace('ò', 'o').replace('ỏ', 'o').replace('õ', 'o').replace('ọ', 'o').replace('ô', 'o').replace('ơ', 'o').replace('ú', 'u').replace('ù', 'u').replace('ủ', 'u').replace('ũ', 'u').replace('ụ', 'u').replace('ư', 'u').replace('ý', 'y').replace('ỳ', 'y').replace('ỷ', 'y').replace('ỹ', 'y').replace('ỵ', 'y').replace(' ', '_')

def extract_clip_features():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    features = []
    for fname in os.listdir(IMG_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        img_path = os.path.join(IMG_DIR, fname)
        # Tìm subject cho ảnh này (normalize)
        subject_found = None
        for subject in SUBJECTS:
            norm_subj = normalize_subject(subject)
            if norm_subj in normalize_subject(fname):
                subject_found = subject
                break
        if not subject_found:
            print(f"Warning: {fname} không xác định được chủ đề!")
            continue
        try:
            image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(image)
            features.append({
                'file': fname,
                'feature': feat.cpu().numpy().flatten(),
                'subject': subject_found
            })
            print(f"Done: {fname} ({subject_found})")
        except Exception as e:
            print(f"Error {fname}: {e}")
    np.save(FEATURE_PATH, features)
    print(f"Saved features to {FEATURE_PATH}")
    # Tạo vector đại diện cho từng chủ đề
    subject_vectors = {}
    for subject in SUBJECTS:
        subject_feats = [f['feature'] for f in features if f['subject'] == subject]
        if subject_feats:
            subject_vectors[subject] = np.mean(subject_feats, axis=0)
    np.save(SUBJECT_PATH, subject_vectors)
    print(f"Saved subject vectors to {SUBJECT_PATH}")

if __name__ == "__main__":
    extract_clip_features()

