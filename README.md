# Central Researcher API - Deployment Package

FAISS + GCS FUSEによる高速論文検索APIのデプロイパッケージです。

## 構成

```
central-researcher-deploy/
├── app.py                     # Flask APIサーバー
├── Dockerfile                 # Dockerビルド設定
├── requirements-api.txt       # Python依存関係
├── deploy.sh                  # デプロイスクリプト
├── firebase.json              # Firebase設定
├── public/                    # フロントエンド
├── article_search/            # FAISS論文検索モジュール
└── researcher_search/         # 研究者分析モジュール
```

## 前提条件

1. Google Cloud SDK (`gcloud`) インストール済み
2. Firebase CLI (`firebase`) インストール済み
3. Docker インストール済み
4. GCSバケット `proj-utokyo-vectors` に以下が配置済み:
   - `faiss_index/papers.index` (FAISSインデックス)
   - `faiss_index/papers_metadata.db` (SQLiteメタデータ)

## デプロイ手順

### 1. Google Cloudプロジェクトを設定

```bash
gcloud config set project YOUR_PROJECT_ID
gcloud auth login
```

### 2. バックエンド（Cloud Run）をデプロイ

```bash
./deploy.sh backend
```

### 3. フロントエンド（Firebase Hosting）をデプロイ

```bash
./deploy.sh frontend
```

### 4. 全てをデプロイ

```bash
./deploy.sh all
```

## 環境変数

| 変数名 | 説明 | デフォルト |
|--------|------|-----------|
| `PORT` | サーバーポート | 8080 |
| `GCS_BUCKET_NAME` | GCSバケット名 | proj-utokyo-vectors |
| `OPENAI_API_KEY` | OpenAI APIキー | 必須 |

## APIエンドポイント

| エンドポイント | メソッド | 説明 |
|---------------|----------|------|
| `/api/health` | GET | ヘルスチェック |
| `/api/analyze` | POST | 論文検索＆研究者分析 |
| `/api/researchers/<id>` | GET | 研究者詳細取得 |

## ローカルテスト

```bash
# 依存関係インストール
pip install -r requirements-api.txt

# 環境変数設定
export OPENAI_API_KEY="your-api-key"

# サーバー起動
python app.py
```

## 注意事項

- FAISSインデックスは起動時にGCSからダウンロードされます（約4.7GB）
- SQLiteメタデータはGCS FUSEマウント経由でアクセスされます
- Cloud Runのメモリは最低2GBが必要です
