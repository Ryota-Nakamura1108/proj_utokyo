#!/bin/bash

# Central Researcher Analysis - Cloud Build版デプロイスクリプト
# Dockerのインストール不要！Google Cloud Buildを使用してデプロイします
# 使用方法: ./deploy_cloudbuilder.sh [backend|frontend|all]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

echo_error() {
    echo -e "${RED}✗ $1${NC}"
}

# プロジェクトIDを取得
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo_error "Google Cloud プロジェクトが設定されていません"
    echo "次のコマンドでプロジェクトを設定してください:"
    echo "gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo_success "Google Cloud プロジェクト: $PROJECT_ID"

# Cloud Build APIの有効化
echo ""
echo "Cloud Build APIを有効化中..."
gcloud services enable cloudbuild.googleapis.com --quiet
echo_success "Cloud Build APIを有効化しました"

# バックエンドのデプロイ
deploy_backend() {
    echo ""
    echo "=========================================="
    echo "バックエンド (Cloud Run) のデプロイ"
    echo "=========================================="

    # ベクトルディレクトリの確認
    if [ ! -d "article_search/vectors" ]; then
        echo_warning "ベクトルディレクトリが見つかりません: article_search/vectors"
        echo_warning "空のディレクトリを作成します..."
        mkdir -p article_search/vectors
    else
        echo_success "ベクトルディレクトリを確認しました"
    fi

    # GCSバケット名を取得
    BUCKET_NAME="${PROJECT_ID}-vectors"

    # OPENAI_API_KEYを.envファイルから取得
    if [ -f .env ]; then
        export $(grep -v '^#' .env | xargs)
    fi

    if [ -z "$OPENAI_API_KEY" ]; then
        echo_error "OPENAI_API_KEY が .env ファイルに設定されていません"
        exit 1
    fi

    echo ""
    echo "Cloud Buildを使用してイメージをビルド＆プッシュ中..."
    echo "（この処理はGoogleのサーバー上で実行されるため、ローカルのDockerは不要です）"
    echo ""

    # Cloud Buildを使用してビルド
    gcloud builds submit --tag gcr.io/$PROJECT_ID/central-researcher-api:latest .

    echo_success "イメージのビルドとプッシュが完了しました"

    # Cloud Runにデプロイ
    echo ""
    echo "Cloud Runにデプロイ中..."
    echo "環境変数 GCS_BUCKET_NAME=$BUCKET_NAME を設定します"
    echo "環境変数 OPENAI_API_KEY を設定します"

    gcloud run deploy central-researcher-api \
        --image gcr.io/$PROJECT_ID/central-researcher-api:latest \
        --platform managed \
        --region asia-northeast1 \
        --allow-unauthenticated \
        --memory 32Gi \
        --cpu 8 \
        --timeout 1800 \
        --max-instances 10 \
        --set-env-vars GCS_BUCKET_NAME=$BUCKET_NAME,OPENAI_API_KEY=$OPENAI_API_KEY,FAISS_INDEX_DIR=/mnt/gcs/faiss_index \
        --add-volume name=faiss-volume,type=cloud-storage,bucket=proj-utokyo-vectors \
        --add-volume-mount volume=faiss-volume,mount-path=/mnt/gcs \
        --quiet

    # URLを取得
    SERVICE_URL=$(gcloud run services describe central-researcher-api \
        --platform managed \
        --region asia-northeast1 \
        --format 'value(status.url)')

    echo_success "バックエンドのデプロイが完了しました"
    echo ""
    echo "=========================================="
    echo "Cloud Run URL: $SERVICE_URL"
    echo "=========================================="
    echo ""
    echo_warning "public/index.html の API_URL を以下に更新してください:"
    echo "const API_URL = '$SERVICE_URL';"
    echo ""
}

# フロントエンドのデプロイ
deploy_frontend() {
    echo ""
    echo "=========================================="
    echo "フロントエンド (Firebase Hosting) のデプロイ"
    echo "=========================================="

    # publicディレクトリの確認
    if [ ! -d "public" ]; then
        echo_error "publicディレクトリが見つかりません"
        exit 1
    fi

    # Firebase Hostingにデプロイ
    echo ""
    echo "Firebase Hostingにデプロイ中..."
    firebase deploy --only hosting

    echo_success "フロントエンドのデプロイが完了しました"
}

# ヘルプメッセージ
show_help() {
    echo "使用方法: ./deploy_cloudbuilder.sh [オプション]"
    echo ""
    echo "オプション:"
    echo "  backend   バックエンド (Cloud Run) のみデプロイ"
    echo "  frontend  フロントエンド (Firebase Hosting) のみデプロイ"
    echo "  all       バックエンドとフロントエンドの両方をデプロイ (デフォルト)"
    echo "  help      このヘルプメッセージを表示"
    echo ""
    echo "特徴:"
    echo "  - Dockerのインストール不要"
    echo "  - Google Cloud Buildを使用してクラウド上でビルド"
    echo "  - ローカルマシンのリソースを消費しない"
    echo ""
}

# メイン処理
case "${1:-all}" in
    backend)
        deploy_backend
        ;;
    frontend)
        deploy_frontend
        ;;
    all)
        deploy_backend
        echo ""
        read -p "フロントエンドもデプロイしますか？ (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            deploy_frontend
        else
            echo_warning "フロントエンドのデプロイをスキップしました"
        fi
        ;;
    help)
        show_help
        ;;
    *)
        echo_error "無効なオプション: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo_success "デプロイが完了しました！"
echo ""
