# 🚀 クイックスタートガイド（Cloud Storage版）

Cloud Storageを使用した低コストデプロイの手順です。

## 💰 コスト比較

| 項目 | 通常版 | Cloud Storage版 |
|------|--------|-----------------|
| Container Registry | $3-4/月 | $0.02/月 |
| Cloud Storage | - | $0.20/月 |
| Cloud Run | 使用量による | 使用量による |
| **合計** | **$3-5/月** | **$0.25-2/月** |

**約90%のコスト削減！**

## 📋 前提条件

- [x] Firebase CLI インストール済み
- [x] Google Cloud SDK (gcloud) インストール済み
- [x] Docker インストール済み
- [x] Firebaseプロジェクト作成済み
- [x] Firebase プロジェクトID設定済み（proj-utokyo）

## ⚡ 4ステップでデプロイ

### ステップ1: プロジェクト設定

```bash
# ディレクトリに移動
cd /Users/nakamuraryouta/Desktop/Python/Mlab/projUT/src/central_researcher

# Google Cloud プロジェクトを設定
gcloud config set project proj-utokyo

# 必要なAPIを有効化
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage.googleapis.com
```

### ステップ2: ベクトルデータをCloud Storageにアップロード

```bash
# セットアップスクリプトを実行
./setup_cloud_storage.sh
```

このスクリプトは以下を自動で行います：
- Cloud Storageバケットの作成（`proj-utokyo-vectors`）
- ベクトルデータのアップロード
- 環境設定ファイルの作成

**所要時間**: 約5-15分（ベクトルデータのサイズによる）

**進捗確認**:
```bash
# アップロード状況を確認
gsutil ls gs://proj-utokyo-vectors/vectors/ | wc -l
# → 94行程度表示されればOK
```

### ステップ3: バックエンドをデプロイ

```bash
# デプロイスクリプトを実行
./deploy.sh backend
```

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

`public/index.html` を開いて、45行目付近を編集:

```javascript
// 変更前
const API_URL = 'http://localhost:5000';

// 変更後（あなたのCloud Run URLに置き換え）
const API_URL = 'https://central-researcher-api-xxxxx-an.a.run.app';
```

#### 4-2. フロントエンドをデプロイ

```bash
# Firebase Hostingにデプロイ
./deploy.sh frontend
```

**所要時間**: 約1-2分

**成功すると以下が表示されます**:
```
✔ Deploy complete!
Hosting URL: https://proj-utokyo.web.app
```

## ✅ 動作確認

### 1. ヘルスチェック

```bash
# Cloud Run URLをテスト
curl https://your-cloud-run-url/api/health
```

**期待される結果**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T...",
  "modules": {
    "PaperSearchEngine": true,
    "CentralResearcher": true,
    ...
  }
}
```

### 2. Webアプリにアクセス

ブラウザで以下にアクセス:
```
https://proj-utokyo.web.app
```

### 3. テスト分析を実行

1. クエリ入力: `quantum computing`
2. メールアドレス入力: あなたのメールアドレス
3. 機関: `The University of Tokyo`
4. 「分析開始」をクリック

**注意**: 初回実行時、Cloud Runが起動してベクトルデータをCloud Storageからダウンロードするため、**5-10分程度**かかります。

## 🎯 完了！

これで以下が利用可能になりました：

- ✅ Webフォーム: `https://proj-utokyo.web.app`
- ✅ API: `https://your-cloud-run-url/api/analyze`
- ✅ 低コスト運用: 月額$0.25-2程度

## 📊 デプロイ後の確認

### Cloud Storageを確認

```bash
# バケットの存在確認
gsutil ls | grep vectors

# ファイル数を確認
gsutil ls gs://proj-utokyo-vectors/vectors/ | wc -l
# → 94程度であればOK

# ストレージ使用量を確認
gsutil du -sh gs://proj-utokyo-vectors/
# → 約10GB程度
```

### Cloud Runの設定を確認

```bash
# サービス情報を表示
gcloud run services describe central-researcher-api \
  --region asia-northeast1

# 環境変数を確認
gcloud run services describe central-researcher-api \
  --region asia-northeast1 \
  --format 'value(spec.template.spec.containers[0].env)'
# → GCS_BUCKET_NAME=proj-utokyo-vectors が設定されているはず
```

## 🔄 更新方法

### ベクトルデータを更新する場合

```bash
# 新しいデータをCloud Storageにアップロード
gsutil -m rsync -r article_search/vectors/ gs://proj-utokyo-vectors/vectors/

# Cloud Runインスタンスをリセット
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0
```

詳しくは `UPDATE_GUIDE.md` を参照してください。

### コードを修正する場合

```bash
# バックエンドのみ再デプロイ
./deploy.sh backend

# フロントエンドのみ再デプロイ
./deploy.sh frontend
```

## 🐛 トラブルシューティング

### エラー1: Cloud Storageバケットが見つからない

```bash
# バケットを手動で作成
gsutil mb -l asia-northeast1 gs://proj-utokyo-vectors

# ベクトルデータをアップロード
./setup_cloud_storage.sh
```

### エラー2: Cloud Runが起動しない

```bash
# ログを確認
gcloud run services logs read central-researcher-api \
  --region asia-northeast1 \
  --limit 50
```

よくあるエラー:
- `GCS_BUCKET_NAME not set`: 環境変数が設定されていない
- `Permission denied`: Cloud Storageへのアクセス権限がない

**解決策**:
```bash
# 環境変数を再設定
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --set-env-vars GCS_BUCKET_NAME=proj-utokyo-vectors
```

### エラー3: 初回実行がタイムアウトする

**原因**: ベクトルデータのダウンロードに時間がかかっている

**解決策**: タイムアウトを延長
```bash
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --timeout 900
```

### エラー4: CORS エラー

**原因**: `public/index.html` のAPI URLが正しくない

**解決策**:
1. Cloud Run URLを確認
   ```bash
   gcloud run services describe central-researcher-api \
     --region asia-northeast1 \
     --format 'value(status.url)'
   ```
2. `public/index.html` を修正
3. 再デプロイ
   ```bash
   ./deploy.sh frontend
   ```

## 💡 コスト削減のヒント

### 1. 使わない時は最小インスタンス数を0に

```bash
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0
```

### 2. 予算アラートを設定

1. [GCP Console](https://console.cloud.google.com/billing) にアクセス
2. Budgets & alerts をクリック
3. 予算額を設定（例: $5/月）

### 3. 定期的に料金を確認

```bash
# Cloud Storageの使用量
gsutil du -sh gs://proj-utokyo-vectors/

# Cloud Runの統計（GCP Console推奨）
# https://console.cloud.google.com/run
```

## 📚 次のステップ

- [ ] カスタムドメインを設定（オプション）
- [ ] 認証機能を追加（Firebase Authentication）
- [ ] モニタリングを設定（Cloud Monitoring）
- [ ] 定期的にベクトルデータを更新

## 🎓 学習リソース

- Cloud Storage ガイド: `DEPLOYMENT_GUIDE.md`
- 更新・メンテナンス: `UPDATE_GUIDE.md`
- API仕様: `API_README.md`

---

## ✨ おめでとうございます！

Cloud Storage版のデプロイが完了しました。低コストで運用できる研究者分析システムをお楽しみください！

何か問題があれば、`UPDATE_GUIDE.md` のトラブルシューティングセクションを参照してください。
