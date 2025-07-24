# DeepVision Search

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.8+ và môi trường ảo:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### 3. Khởi động server
```bash
python3 server.py
```

Sau đó mở trình duyệt truy cập: [http://127.0.0.1:5001](http://127.0.0.1:5001)

### 4. Cấu trúc dự án
- `server.py`: Chạy web app tìm kiếm ảnh
- `feature_extractor.py`: Trích xuất features cho ảnh mới
- `static/clip_features.npy`, `static/clip_subjects.npy`: Dữ liệu vector ảnh
- `static/img/`: Thư mục chứa ảnh
- `templates/`: Giao diện HTML
- `requirements.txt`: Danh sách thư viện

### 5. Lưu ý
- Để thêm ảnh mới: copy vào `static/img/` rồi chạy lại `feature_extractor.py`
- Đảm bảo các file .npy và ảnh không bị xóa khi backup/dọn dẹp
- Có thể nén toàn bộ thư mục dự án để lưu trữ hoặc chuyển sang máy khác

### 6. Khởi động nhanh (script)
Có thể dùng file `start.sh` để tự động kích hoạt môi trường và chạy server:

```bash
#!/bin/bash
source venv/bin/activate
python3 server.py
```

Chạy:
```bash
bash start.sh
```

## Tự động chạy server khi khởi động máy (macOS)

1. Mở Terminal, gõ:
   ```sh
   crontab -e
   ```
2. Thêm dòng sau vào cuối file:
   ```sh
   @reboot cd /Users/macbook/Downloads/sis && /bin/zsh start.sh
   ```
   Hoặc nếu chạy bằng python:
   ```sh
   @reboot cd /Users/macbook/Downloads/sis && /usr/bin/python3 server.py
   ```

## Chạy server nền (không tắt khi đóng Terminal)

```sh
nohup ./start.sh &
```
hoặc
```sh
nohup python server.py &
```


