# 🚀 クイックスタートガイド（Docker不要版）

**Dockerのインストール不要！** Google Cloud Buildを使用してデプロイします。

## ✅ 必要なもの

- [x] Firebase CLI (`npm install -g firebase-tools`)
- [x] Google Cloud SDK (`gcloud`)
- [ ] ~~Docker~~ **不要！**
- [x] Firebaseプロジェクト（proj-utokyo）

## ⚡ 4ステップでデプロイ

### ステップ1: プロジェクト設定と必要なAPIの有効化

```bash
# ディレクトリに移動
cd /Users/nakamuraryouta/Desktop/Python/Mlab/projUT/src/central_researcher

# Google Cloud プロジェクトを設定
gcloud config set project proj-utokyo

# 必要なAPIを有効化
gcloud services enable run.googleapis.com \
  containerregistry.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com
```

### ステップ2: ベクトルデータをCloud Storageにアップロード

```bash
# セットアップスクリプトを実行
./setup_cloud_storage.sh
```

**進捗**:
- バケット作成: `proj-utokyo-vectors`
- ファイル数: 94個
- 所要時間: 5-15分

**確認**:
```bash
gsutil ls gs://proj-utokyo-vectors/vectors/ | wc -l
# → 94行程度表示されればOK
```

### ステップ3: バックエンドをデプロイ（Docker不要）

```bash
# Cloud Build版デプロイスクリプトを実行
./deploy_cloudbuilder.sh backend
```

**特徴**:
- ✅ Dockerのインストール不要
- ✅ Google Cloud Buildがクラウド上でビルド
- ✅ ローカルマシンのリソースを消費しない

**所要時間**: 約10-15分（初回）

**成功すると以下が表示されます**:
```
========================================
Cloud Run URL: https://central-researcher-api-xxxxx-an.a.run.app
========================================
```

**このURLをメモしてください！**

### ステップ4: フロントエンドを設定してデプロイ

#### 4-1. API URLを更新

エディタで `public/index.html` を開いて、45行目付近を編集:

```bash
# エディタで開く（お好みのエディタを使用）
open -e public/index.html
# または
vim public/index.html
# または
nano public/index.html
```

**変更箇所**（45行目付近）:
```javascript
// 変更前
const API_URL = 'http://localhost:5000';

// 変更後（ステップ3で表示されたCloud Run URLに置き換え）
const API_URL = 'https://central-researcher-api-xxxxx-an.a.run.app';
```

#### 4-2. フロントエンドをデプロイ

```bash
# Firebase Hostingにデプロイ
./deploy_cloudbuilder.sh frontend
```

**所要時間**: 約1-2分

**成功すると以下が表示されます**:
```
✔ Deploy complete!
Hosting URL: https://proj-utokyo.web.app
```

## ✅ 動作確認

### 1. APIヘルスチェック

```bash
# Cloud Run URLを確認（ステップ3でメモしたURL）
curl https://your-cloud-run-url/api/health
```

**期待される結果**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T...",
  "modules": {
    "PaperSearchEngine": true,
    "CentralResearcher": true
  }
}
```

### 2. Webアプリにアクセス

ブラウザで開く:
```
https://proj-utokyo.web.app
```

### 3. テスト分析を実行

1. **クエリ**: `quantum computing`
2. **メールアドレス**: あなたのメールアドレス
3. **機関**: `The University of Tokyo`
4. 「**分析開始**」をクリック

**注意**: 初回実行は5-10分程度かかります（ベクトルデータのダウンロード）。

## 🎯 完了！

これで以下が利用可能になりました：

- ✅ Webアプリ: `https://proj-utokyo.web.app`
- ✅ API: `https://your-cloud-run-url/api/analyze`
- ✅ 低コスト運用: 月額$0.25-2程度

## 💰 コスト目安

| 項目 | 月額料金 |
|------|---------|
| Cloud Storage | $0.20 |
| Container Registry | $0.02 |
| Cloud Run | 使用量による（無料枠内なら$0） |
| Cloud Build | 初回120分無料、以降$0.003/分 |
| **合計** | **$0.25-2** |

## 🔄 更新方法

### ベクトルデータを更新

```bash
# Cloud Storageに新しいデータをアップロード
gsutil -m rsync -r article_search/vectors/ gs://proj-utokyo-vectors/vectors/

# インスタンスをリセット
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0
```

### コードを修正

```bash
# コードを編集
vim app.py

# 再デプロイ
./deploy_cloudbuilder.sh backend
```

## 🐛 トラブルシューティング

### エラー1: Cloud Build APIが無効

```bash
# APIを有効化
gcloud services enable cloudbuild.googleapis.com
```

### エラー2: 権限エラー

```bash
# Cloud Buildサービスアカウントに権限を付与
PROJECT_NUMBER=$(gcloud projects describe proj-utokyo --format='value(projectNumber)')

gcloud projects add-iam-policy-binding proj-utokyo \
  --member=serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com \
  --role=roles/run.admin

gcloud projects add-iam-policy-binding proj-utokyo \
  --member=serviceAccount:${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com \
  --role=roles/iam.serviceAccountUser
```

### エラー3: ビルドがタイムアウト

```bash
# タイムアウトを延長してリトライ
gcloud builds submit --timeout=1800s --tag gcr.io/proj-utokyo/central-researcher-api:latest .
```

### エラー4: Cloud Storageバケットが見つからない

```bash
# バケットを確認
gsutil ls | grep vectors

# なければステップ2を再実行
./setup_cloud_storage.sh
```

## 📊 Cloud Buildの特徴

### メリット
- ✅ Dockerのインストール不要
- ✅ ローカルマシンのリソース消費なし
- ✅ 高速ビルド（Googleのインフラ使用）
- ✅ 自動的にキャッシュを利用

### デメリット
- ⚠️ ビルド時間に応じた課金（初回120分無料）
- ⚠️ インターネット接続が必要

## 💡 ヒント

### Cloud Buildの料金を確認

```bash
# ビルド履歴を確認
gcloud builds list --limit=10

# 特定のビルドの詳細
gcloud builds describe BUILD_ID
```

### ビルドログを確認

```bash
# 最新のビルドログを表示
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')
```

## 🎓 次のステップ

1. **料金アラートを設定**
   - [GCP Console](https://console.cloud.google.com/billing) → Budgets & alerts

2. **定期的にログを確認**
   ```bash
   gcloud run services logs read central-researcher-api \
     --region asia-northeast1 \
     --limit 50
   ```

3. **ベクトルデータを定期的に更新**
   - 新しい論文データを追加
   - Cloud Storageに同期

## 📚 関連ドキュメント

- **更新・メンテナンス**: `UPDATE_GUIDE.md`
- **API仕様**: `API_README.md`
- **詳細なデプロイ手順**: `DEPLOYMENT_GUIDE.md`

---

## ✨ おめでとうございます！

Dockerなしでデプロイが完了しました。お疲れ様でした！🎉

何か問題があれば、このガイドのトラブルシューティングセクションを参照してください。
