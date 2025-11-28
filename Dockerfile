# Python 3.11ベースイメージを使用
FROM python:3.11-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージの更新とインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 依存関係ファイルをコピー
COPY requirements-api.txt .

# Pythonパッケージをインストール
RUN pip install --no-cache-dir -r requirements-api.txt

# アプリケーションコードをコピー（public/ディレクトリを含む）
COPY . .

# メインのrequirements.txtが存在する場合はインストール
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# データディレクトリを作成
RUN mkdir -p /app/output
RUN mkdir -p /app/article_search/vectors

# ポート8080を公開（Cloud Runのデフォルト）
EXPOSE 8080

# 環境変数を設定
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DOWNLOAD_VECTORS_ON_STARTUP=false

# GCS_BUCKET_NAME は Cloud Run デプロイ時に設定される

# gunicornでアプリケーションを起動
# --preload: Load app code before worker processes are forked (prevents multiple FAISS loads)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 1200 --preload app:app
