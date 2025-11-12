#!/bin/bash

# Cloud Storage セットアップスクリプト
# ベクトルデータをCloud Storageにアップロードします

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
    exit 1
fi

echo_success "プロジェクトID: $PROJECT_ID"

# バケット名を設定
BUCKET_NAME="${PROJECT_ID}-vectors"
REGION="asia-northeast1"

echo ""
echo "=========================================="
echo "Cloud Storage セットアップ"
echo "=========================================="
echo "バケット名: gs://${BUCKET_NAME}"
echo "リージョン: ${REGION}"
echo ""

# Storage APIの有効化
echo "Cloud Storage APIを有効化中..."
gcloud services enable storage.googleapis.com
echo_success "Cloud Storage APIを有効化しました"

# バケットの存在確認
if gsutil ls -b gs://${BUCKET_NAME} >/dev/null 2>&1; then
    echo_warning "バケット gs://${BUCKET_NAME} は既に存在します"
else
    # バケットを作成
    echo "バケットを作成中..."
    gsutil mb -l ${REGION} gs://${BUCKET_NAME}
    echo_success "バケットを作成しました: gs://${BUCKET_NAME}"
fi

# ベクトルディレクトリの確認
VECTORS_DIR="article_search/vectors"
if [ ! -d "$VECTORS_DIR" ]; then
    echo_error "ベクトルディレクトリが見つかりません: $VECTORS_DIR"
    exit 1
fi

# ファイル数を確認
FILE_COUNT=$(ls -1 $VECTORS_DIR/*.h5 $VECTORS_DIR/*.pkl 2>/dev/null | wc -l)
echo_success "ベクトルファイル数: ${FILE_COUNT}個"

# アップロード確認
echo ""
echo_warning "以下のファイルをCloud Storageにアップロードします:"
echo "  - ソース: $VECTORS_DIR"
echo "  - 送信先: gs://${BUCKET_NAME}/vectors/"
echo "  - ファイル数: ${FILE_COUNT}個"
echo "  - 推定時間: 5-15分"
echo ""
read -p "続行しますか？ (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo_warning "アップロードをキャンセルしました"
    exit 0
fi

# アップロード開始
echo ""
echo "ベクトルデータをアップロード中..."
echo "（進捗状況が表示されます）"
echo ""

gsutil -m rsync -r -d $VECTORS_DIR gs://${BUCKET_NAME}/vectors/

echo ""
echo_success "アップロードが完了しました！"

# アップロードされたファイルを確認
echo ""
echo "アップロードされたファイルを確認中..."
UPLOADED_COUNT=$(gsutil ls gs://${BUCKET_NAME}/vectors/ | wc -l)
echo_success "Cloud Storage上のファイル数: ${UPLOADED_COUNT}個"

# バケット情報を表示
echo ""
echo "=========================================="
echo "セットアップ完了"
echo "=========================================="
echo "バケット名: gs://${BUCKET_NAME}"
echo "パス: gs://${BUCKET_NAME}/vectors/"
echo ""
echo_success "次のステップ: ./deploy.sh all でデプロイを実行"
echo ""

# .env ファイルに環境変数を書き出し（オプション）
echo "GCS_BUCKET_NAME=${BUCKET_NAME}" > .env.gcs
echo_success ".env.gcs ファイルを作成しました"
echo ""
echo "バケット名を確認: cat .env.gcs"
