import os
from duckduckgo_search import DDGS
import requests
from PIL import Image
from io import BytesIO
import glob

# Danh sách chủ đề phổ biến
TOPICS = [
    'cá', 'chó', 'mèo', 'biển', 'hoa', 'xe', 'gấu', 'rừng', 'sách', 'thời trang', 'tranh', 'báo', 'nhà', 'rạp phim', 'vịt', 'IT'
]

SAVE_DIR = 'static/img/'
IMAGES_PER_TOPIC = 3

os.makedirs(SAVE_DIR, exist_ok=True)

def count_images_for_topic(topic):
    # Đếm số file có tên chứa chủ đề (không phân biệt hoa thường/hoa hoa)
    files = glob.glob(os.path.join(SAVE_DIR, f"*{topic.replace(' ', '_')}*.jpg"))
    return len(files)

def download_images(query, n=3, skip_existing=True):
    existing = count_images_for_topic(query)
    need = max(0, n - existing)
    if need == 0:
        print(f"Đã đủ ảnh cho chủ đề: {query}")
        return
    with DDGS() as ddgs:
        results = ddgs.images(query, max_results=need*2)  # Lấy dư để tránh trùng
        downloaded = 0
        for r in results:
            url = r['image']
            try:
                resp = requests.get(url, timeout=10)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                # Đặt tên không trùng
                idx = 1
                while True:
                    filename = f"{query.replace(' ', '_')}_{existing+idx}.jpg"
                    path = os.path.join(SAVE_DIR, filename)
                    if not os.path.exists(path):
                        break
                    idx += 1
                img.save(path)
                print(f"Đã tải: {path}")
                downloaded += 1
                if downloaded >= need:
                    break
            except Exception as e:
                print(f"Lỗi tải {url}: {e}")

if __name__ == '__main__':
    for topic in TOPICS:
        print(f"--- Kiểm tra chủ đề: {topic} ---")
        download_images(topic, IMAGES_PER_TOPIC) 
        