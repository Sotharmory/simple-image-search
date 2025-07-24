import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
import webbrowser
import threading
import clip
import torch
import os
import hashlib
import re
import json
from datetime import datetime
from collections import Counter
import faiss
# XÓA: from googletrans import Translator
import unicodedata

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

LOG_FILE = 'search_logs.jsonl'

def log_search(query_type, value):
    log_entry = {
        'type': query_type,
        'value': value,
        'time': datetime.now().isoformat(timespec='seconds')
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

app = Flask(__name__)

# Load model và features CLIP
clip_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=clip_device)
clip_features = np.load('static/clip_features.npy', allow_pickle=True)
clip_subjects = np.load('static/clip_subjects.npy', allow_pickle=True).item()

# Chuẩn bị FAISS index cho clip_features
clip_vectors = np.stack([item['feature'] for item in clip_features])
faiss_index = faiss.IndexFlatL2(clip_vectors.shape[1])
faiss_index.add(clip_vectors)
file_names = [item['file'] for item in clip_features]

# Cache kết quả truy vấn theo hash ảnh
query_cache = {}

from numpy.linalg import norm
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def normalize_subject(s):
    return s.lower().replace('đ', 'd').replace('á', 'a').replace('à', 'a').replace('ả', 'a').replace('ã', 'a').replace('ạ', 'a').replace('â', 'a').replace('ă', 'a').replace('é', 'e').replace('è', 'e').replace('ẻ', 'e').replace('ẽ', 'e').replace('ẹ', 'e').replace('ê', 'e').replace('í', 'i').replace('ì', 'i').replace('ỉ', 'i').replace('ĩ', 'i').replace('ị', 'i').replace('ó', 'o').replace('ò', 'o').replace('ỏ', 'o').replace('õ', 'o').replace('ọ', 'o').replace('ô', 'o').replace('ơ', 'o').replace('ú', 'u').replace('ù', 'u').replace('ủ', 'u').replace('ũ', 'u').replace('ụ', 'u').replace('ư', 'u').replace('ý', 'y').replace('ỳ', 'y').replace('ỷ', 'y').replace('ỹ', 'y').replace('ỵ', 'y').replace(' ', '_')

def normalize_text(s):
    s = s.lower()
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    s = s.replace('đ', 'd')
    s = s.replace(' ', '_')
    return s

# Subject thực tế trong feature
SUBJECTS = ['IT', 'biển', 'báo', 'chó', 'cá', 'gấu', 'hoa', 'mèo', 'nhà', 'rạp_phim', 'rừng', 'sách', 'thời_trang', 'tranh', 'vịt', 'xe']

# Mapping synonym cho subject: mọi biến thể từ khóa về subject chuẩn
SUBJECT_SYNONYMS = {
    # IT
    'it': 'IT', 'information_technology': 'IT', 'cong_nghe_thong_tin': 'IT', 'cntt': 'IT', 'tech': 'IT', 'technology': 'IT',
    # biển
    'bien': 'biển', 'sea': 'biển', 'bai_bien': 'biển', 'beach': 'biển', 'ocean': 'biển', 'biển': 'biển', 'bienxanh': 'biển',
    # báo
    'bao': 'báo', 'tiger': 'báo', 'con_bao': 'báo', 'ho_bao': 'báo', 'ho': 'báo', 'bengal': 'báo', 'leopard': 'báo', 'panther': 'báo',
    # chó
    'cho': 'chó', 'dog': 'chó', 'con_cho': 'chó', 'puppy': 'chó', 'doggy': 'chó', 'cún': 'chó', 'pet_dog': 'chó', 'chihuahua': 'chó', 'husky': 'chó', 'poodle': 'chó',
    # cá
    'ca': 'cá', 'fish': 'cá', 'con_ca': 'cá', 'fishes': 'cá', 'goldfish': 'cá', 'betta': 'cá', 'catfish': 'cá', 'carp': 'cá',
    # gấu
    'gau': 'gấu', 'bear': 'gấu', 'con_gau': 'gấu', 'polar_bear': 'gấu', 'brown_bear': 'gấu', 'black_bear': 'gấu', 'grizzly': 'gấu',
    # hoa
    'hoa': 'hoa', 'flower': 'hoa', 'bong_hoa': 'hoa', 'flowers': 'hoa', 'rose': 'hoa', 'tulip': 'hoa', 'orchid': 'hoa', 'sunflower': 'hoa', 'daisy': 'hoa', 'hoa_hong': 'hoa',
    # mèo
    'meo': 'mèo', 'meow': 'mèo', 'cat': 'mèo', 'con_meo': 'mèo', 'kitten': 'mèo', 'kitty': 'mèo', 'pet_cat': 'mèo', 'tabby': 'mèo', 'persian': 'mèo',
    # nhà
    'nha': 'nhà', 'house': 'nhà', 'home': 'nhà', 'casa': 'nhà', 'building': 'nhà', 'villa': 'nhà', 'apartment': 'nhà', 'cottage': 'nhà',
    # rạp phim
    'rap_phim': 'rạp_phim', 'cinema': 'rạp_phim', 'movie_theater': 'rạp_phim', 'rap': 'rạp_phim', 'theater': 'rạp_phim', 'phong_chieu': 'rạp_phim', 'phim': 'rạp_phim',
    # rừng
    'rung': 'rừng', 'forest': 'rừng', 'khu_rung': 'rừng', 'woods': 'rừng', 'jungle': 'rừng', 'rainforest': 'rừng', 'rung_nhiet_doi': 'rừng',
    # sách
    'sach': 'sách', 'book': 'sách', 'books': 'sách', 'novel': 'sách', 'textbook': 'sách', 'truyen': 'sách', 'ebook': 'sách', 'reading': 'sách',
    # thời trang
    'thoi_trang': 'thời_trang', 'fashion': 'thời_trang', 'ao_quan': 'thời_trang', 'clothes': 'thời_trang', 'outfit': 'thời_trang', 'style': 'thời_trang', 'dress': 'thời_trang', 'model': 'thời_trang',
    # tranh
    'tranh': 'tranh', 'painting': 'tranh', 'picture': 'tranh', 'art': 'tranh', 'canvas': 'tranh', 'artwork': 'tranh', 'drawing': 'tranh', 'sketch': 'tranh',
    # vịt
    'vit': 'vịt', 'duck': 'vịt', 'con_vit': 'vịt', 'duckling': 'vịt', 'mallard': 'vịt', 'goose': 'vịt', 'quack': 'vịt',
    # xe
    'xe': 'xe', 'car': 'xe', 'oto': 'xe', 'automobile': 'xe', 'vehicle': 'xe', 'truck': 'xe', 'bus': 'xe', 'motorbike': 'xe', 'bike': 'xe', 'bicycle': 'xe', 'scooter': 'xe',
}

# Bỏ secret_key và session

# Bỏ các route /login, /logout và decorator login_required

# Chuỗi giao diện đa ngôn ngữ
LANGS = {
    'vi': {
        'brand': 'DeepVision Search',
        'about': 'Giới thiệu & Hướng dẫn',
        'search_btn': 'Tìm kiếm',
        'search_text_btn': 'Tìm kiếm bằng văn bản',
        'input_placeholder': 'Nhập mô tả bằng tiếng Việt hoặc tiếng Anh...',
        'logout': 'Đăng xuất',
        'results_title': 'Kết quả tương tự',
        'no_results': 'Không tìm thấy kết quả phù hợp.',
        'error_no_file': 'Không có file ảnh!',
        'error_no_text': 'Vui lòng nhập mô tả!',
        'about_title': 'Giới thiệu | DeepVision Search',
    },
    'en': {
        'brand': 'DeepVision Search',
        'about': 'About & Guide',
        'search_btn': 'Search',
        'search_text_btn': 'Text Search',
        'input_placeholder': 'Enter a description in English or Vietnamese...',
        'logout': 'Logout',
        'results_title': 'Similar Results',
        'no_results': 'No matching results found.',
        'error_no_file': 'No image file!',
        'error_no_text': 'Please enter a description!',
        'about_title': 'About | DeepVision Search',
    }
}

def get_lang():
    lang = request.args.get('lang', 'vi')
    if lang not in LANGS:
        lang = 'vi'
    return lang

@app.route('/', methods=['GET'])
def index():
    lang = get_lang()
    return render_template('index.html', lang=lang, t=LANGS[lang])

@app.route('/search', methods=['POST'])
def search():
    lang = get_lang()
    file = request.files.get('query_img')
    text = request.form.get('query_text', '').strip()
    if (not file or file.filename == '') and not text:
        return render_template('index.html', error=LANGS[lang]['error_no_file'], lang=lang, t=LANGS[lang])
    # Nếu chỉ có mô tả, chuyển sang search_text
    if (not file or file.filename == '') and text:
        from flask import Request
        with app.test_request_context('/search_text', method='POST', data={'query_text': text}):
            return search_text()
    img_path = os.path.join('static', 'img', 'query_tmp.jpg')
    file.save(img_path)
    file.seek(0)
    file_bytes = file.read()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    log_search('image', file.filename)
    if file_hash in query_cache and not text:
        top_results = query_cache[file_hash]
        return render_template('index.html', query_path=img_path, scores=top_results, lang=lang, t=LANGS[lang])
    image = clip_preprocess(Image.open(img_path)).unsqueeze(0).to(clip_device)
    with torch.no_grad():
        img_feat = clip_model.encode_image(image).cpu().numpy().flatten().astype('float32')
    if text:
        with torch.no_grad():
            text_tokens = clip.tokenize([text]).to(clip_device)
            text_feat = clip_model.encode_text(text_tokens).cpu().numpy().flatten().astype('float32')
        query_feat = (img_feat + text_feat) / 2
    else:
        query_feat = img_feat
    # Xác định chủ đề gần nhất
    best_subject = None
    best_sim = -1
    for subject, subj_vec in clip_subjects.items():
        sim = cosine_similarity(query_feat, subj_vec)
        if sim > best_sim:
            best_sim = sim
            best_subject = subject
    # Lọc ảnh theo subject (không dùng regex tên file)
    filtered_idx = [i for i, f in enumerate(clip_features) if f.get('subject','') == best_subject]
    if filtered_idx:
        sub_vectors = clip_vectors[filtered_idx]
        sub_file_names = [file_names[i] for i in filtered_idx]
        sub_index = faiss.IndexFlatL2(sub_vectors.shape[1])
        sub_index.add(sub_vectors)
        D, I = sub_index.search(query_feat.reshape(1, -1), min(20, len(sub_file_names)))
        top_results = [(sub_file_names[i], float(1 - D[0][j]/2)) for j, i in enumerate(I[0])]
        query_cache[file_hash] = top_results
        return render_template('index.html', query_path=img_path, scores=top_results, lang=lang, t=LANGS[lang])
    # Fallback: tìm toàn bộ ảnh theo vector CLIP
    D, I = faiss_index.search(query_feat.reshape(1, -1), 20)
    top_results = [(file_names[i], float(1 - D[0][j]/2)) for j, i in enumerate(I[0])]
    if top_results:
        return render_template('index.html', query_path=img_path, scores=top_results, lang=lang, t=LANGS[lang])
    # Fallback cuối: trả về ảnh ngẫu nhiên
    import random
    random_files = random.sample(file_names, min(8, len(file_names)))
    top_results = [(fname, 0.0) for fname in random_files]
    return render_template('index.html', query_path=img_path, scores=top_results, lang=lang, t=LANGS[lang])

@app.route('/search_text', methods=['POST'])
def search_text():
    lang = get_lang()
    query = request.form.get('query_text', '').strip()
    if not query:
        return render_template('index.html', error=LANGS[lang]['error_no_text'], lang=lang, t=LANGS[lang])
    log_search('text', query)
    norm_query = normalize_text(query)
    print(f'--- Search query: {query}')
    print(f'--- Normalized query: {norm_query}')
    matched_subject = SUBJECT_SYNONYMS.get(norm_query)
    print(f'--- Matched subject by synonym: {matched_subject}')
    filtered_idx = []
    if matched_subject:
        filtered_idx = [i for i, f in enumerate(clip_features) if normalize_text(f.get('subject','')) == normalize_text(matched_subject)]
        print(f'--- Filtered images: {len(filtered_idx)}')
    if filtered_idx:
        sub_vectors = clip_vectors[filtered_idx]
        sub_file_names = [file_names[i] for i in filtered_idx]
        sub_index = faiss.IndexFlatL2(sub_vectors.shape[1])
        sub_index.add(sub_vectors)
        with torch.no_grad():
            text_tokens = clip.tokenize([matched_subject]).to(clip_device)
            text_feat = clip_model.encode_text(text_tokens).cpu().numpy().flatten().astype('float32')
        D, I = sub_index.search(text_feat.reshape(1, -1), min(20, len(sub_file_names)))
        results = [(sub_file_names[i], float(1 - D[0][j]/2)) for j, i in enumerate(I[0])]
        print(f'--- Final results: {results[:3]} ...')
        return render_template('index.html', text_query=query, scores=results, lang=lang, t=LANGS[lang])
    # Nếu không có subject synonym, tìm subject gần nhất bằng CLIP
    print('--- No synonym match, finding closest subject by CLIP...')
    with torch.no_grad():
        text_tokens = clip.tokenize([query]).to(clip_device)
        text_feat = clip_model.encode_text(text_tokens).cpu().numpy().flatten().astype('float32')
    best_subject = None
    best_sim = -1
    for subject, subj_vec in clip_subjects.items():
        sim = cosine_similarity(text_feat, subj_vec)
        print(f'--- Similarity to subject {subject}: {sim:.4f}')
        if sim > best_sim:
            best_sim = sim
            best_subject = subject
    print(f'--- Best subject by CLIP: {best_subject} (sim={best_sim:.4f})')
    filtered_idx = [i for i, f in enumerate(clip_features) if f.get('subject','') == best_subject]
    print(f'--- Filtered images by best subject: {len(filtered_idx)}')
    if filtered_idx:
        sub_vectors = clip_vectors[filtered_idx]
        sub_file_names = [file_names[i] for i in filtered_idx]
        sub_index = faiss.IndexFlatL2(sub_vectors.shape[1])
        sub_index.add(sub_vectors)
        D, I = sub_index.search(text_feat.reshape(1, -1), min(20, len(sub_file_names)))
        results = [(sub_file_names[i], float(1 - D[0][j]/2)) for j, i in enumerate(I[0])]
        print(f'--- Final results by best subject: {results[:3]} ...')
        return render_template('index.html', text_query=query, scores=results, lang=lang, t=LANGS[lang])
    # Fallback: tìm toàn bộ ảnh theo vector CLIP
    D, I = faiss_index.search(text_feat.reshape(1, -1), 20)
    results = [(file_names[i], float(1 - D[0][j]/2)) for j, i in enumerate(I[0])]
    if results:
        return render_template('index.html', text_query=query, scores=results, lang=lang, t=LANGS[lang])
    # Fallback cuối: trả về ảnh ngẫu nhiên
    import random
    random_files = random.sample(file_names, min(8, len(file_names)))
    results = [(fname, 0.0) for fname in random_files]
    return render_template('index.html', text_query=query, scores=results, lang=lang, t=LANGS[lang])

@app.route('/about', methods=['GET'])
def about():
    lang = get_lang()
    return render_template('about.html', lang=lang, t=LANGS[lang])

@app.route('/admin')
def admin():
    logs = []
    try:
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                logs.append(json.loads(line))
    except FileNotFoundError:
        logs = []
    # Thống kê truy vấn text phổ biến
    text_queries = [log['value'] for log in logs if log['type'] == 'text']
    image_queries = [log['value'] for log in logs if log['type'] == 'image']
    top_text = Counter(text_queries).most_common(10)
    top_image = Counter(image_queries).most_common(10)
    return render_template('admin.html', top_text=top_text, top_image=top_image, logs=logs)

@app.route('/image/<filename>')
def image_detail(filename):
    lang = get_lang()
    # Tìm thông tin ảnh trong clip_features
    info = None
    for item in clip_features:
        if item['file'] == filename:
            info = item
            break
    return render_template('image_detail.html', filename=filename, info=info, lang=lang, t=LANGS[lang])

@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

def open_browser():
    webbrowser.get('safari').open_new("http://127.0.0.1:5001/")

if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(host="0.0.0.0", port=5001, debug=True)
