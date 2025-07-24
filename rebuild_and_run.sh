#!/bin/zsh
cd /Users/macbook/Downloads/sis

# Cài đặt lại requirements nếu cần
python3 -m pip install -r requirements.txt

# Dừng server cũ nếu đang chạy
pkill -f server.py
sleep 2

# Khởi động lại server ở chế độ nền
nohup python3 server.py & 