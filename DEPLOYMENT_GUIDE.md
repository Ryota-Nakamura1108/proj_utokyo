# Firebase Hosting + Cloud Run デプロイガイド

このガイドでは、Central Researcher Analysis システムをFirebase HostingとCloud Runにデプロイする手順を説明します。

## 📋 前提条件

- Firebaseにログイン済み（✅ 完了済み）
- Google Cloud Platform (GCP) のプロジェクトが作成済み
- 以下のツールがインストール済み:
  - Firebase CLI
  - Google Cloud SDK (gcloud)
  - Docker

## 🏗️ アーキテクチャ

```
ユーザー
  ↓
Firebase Hosting (フロントエンド: index.html, results.html)
  ↓
Cloud Run (バックエンド: Flask API)
  ↓
OpenAlex API / ローカルベクトルデータ
```

---

## 📦 ステップ1: 初期設定

### 1.1 Firebaseプロジェクトの設定

```bash
# central_researcherディレクトリに移動
cd /Users/nakamuraryouta/Desktop/Python/Mlab/projUT/src/central_researcher

# Firebase初期化（既に firebase login 済みの場合）
firebase init
```

**初期化時の選択:**
- ✓ Hosting: Configure files for Firebase Hosting
- プロジェクトを選択（既存のプロジェクトを選択）
- Public directory: `public` を入力
- Configure as SPA: `No` を選択
- Set up automatic builds: `No` を選択

### 1.2 .firebasercの更新

`.firebaserc` ファイルを開いて、あなたのFirebaseプロジェクトIDに変更してください:

```json
{
  "projects": {
    "default": "your-actual-firebase-project-id"
  }
}
```

**プロジェクトIDの確認方法:**
```bash
firebase projects:list
```

### 1.3 Google Cloud プロジェクトの確認

```bash
# 現在のプロジェクトを確認
gcloud config get-value project

# プロジェクトを設定（必要に応じて）
gcloud config set project YOUR_PROJECT_ID
```

---

## 🐳 ステップ2: Cloud Runへのバックエンドデプロイ

### 2.1 Google Cloud APIs の有効化

```bash
# Cloud Run APIを有効化
gcloud services enable run.googleapis.com

# Container Registry APIを有効化
gcloud services enable containerregistry.googleapis.com

# Artifact Registry APIを有効化
gcloud services enable artifactregistry.googleapis.com
```

### 2.2 ベクトルデータの準備

**重要**: `article_search/vectors` ディレクトリにベクトルデータが存在することを確認してください。

```bash
# ベクトルディレクトリの確認
ls -la article_search/vectors/
```

ベクトルデータが別の場所にある場合は、コピーしてください:
```bash
# 例: 別の場所からコピー
cp -r /path/to/your/vectors/* article_search/vectors/
```

### 2.3 Dockerイメージのビルドとプッシュ

```bash
# プロジェクトIDを環境変数に設定
export PROJECT_ID=$(gcloud config get-value project)

# Dockerイメージをビルド
docker build -t gcr.io/$PROJECT_ID/central-researcher-api:latest .

# イメージをGoogle Container Registryにプッシュ
docker push gcr.io/$PROJECT_ID/central-researcher-api:latest
```

**注意**: 初回ビルドには数分かかる場合があります。

### 2.4 Cloud Runへのデプロイ

```bash
gcloud run deploy central-researcher-api \
  --image gcr.io/$PROJECT_ID/central-researcher-api:latest \
  --platform managed \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 600 \
  --max-instances 10
```

**パラメータの説明:**
- `--memory 2Gi`: メモリを2GBに設定（分析処理に必要）
- `--cpu 2`: CPUを2コアに設定
- `--timeout 600`: タイムアウトを10分に設定（長時間の分析に対応）
- `--max-instances 10`: 最大インスタンス数を10に設定
- `--allow-unauthenticated`: 認証なしでアクセス可能にする

デプロイが完了すると、Cloud RunのURLが表示されます:
```
Service URL: https://central-researcher-api-xxxxx-an.a.run.app
```

**このURLをメモしてください！**

### 2.5 Cloud Run URLの確認

```bash
# デプロイされたサービスのURLを確認
gcloud run services describe central-researcher-api \
  --platform managed \
  --region asia-northeast1 \
  --format 'value(status.url)'
```

---

## 🌐 ステップ3: フロントエンドの設定

### 3.1 index.htmlのAPI URL更新

`public/index.html` を開いて、API URLを更新します:

```javascript
// 以下の行を見つけて、Cloud RunのURLに変更
const API_URL = 'http://localhost:5000';  // ← この行を変更

// 変更後（あなたのCloud Run URLに置き換えてください）
const API_URL = 'https://central-researcher-api-xxxxx-an.a.run.app';
```

**変更箇所:** `public/index.html` の約45行目

---

## 🚀 ステップ4: Firebase Hostingへのデプロイ

### 4.1 デプロイ

```bash
# Firebase Hostingにデプロイ
firebase deploy --only hosting
```

デプロイが完了すると、Hosting URLが表示されます:
```
✔ Deploy complete!

Hosting URL: https://your-project-id.web.app
```

### 4.2 動作確認

1. ブラウザで `https://your-project-id.web.app` にアクセス
2. クエリ入力フォームが表示されることを確認
3. テストクエリを実行:
   - Query: `quantum computing`
   - Email: あなたのメールアドレス
   - Institution: `The University of Tokyo`
4. 分析が実行され、結果が表示されることを確認

---

## 🔧 トラブルシューティング

### エラー1: ベクトルディレクトリが見つからない

**症状**: API実行時に「Vector directory not found」エラー

**解決策**:
```bash
# ベクトルディレクトリを確認
ls -la article_search/vectors/

# ベクトルデータがない場合は、事前に準備してから再デプロイ
# Dockerイメージを再ビルドして再デプロイ
docker build -t gcr.io/$PROJECT_ID/central-researcher-api:latest .
docker push gcr.io/$PROJECT_ID/central-researcher-api:latest
gcloud run deploy central-researcher-api \
  --image gcr.io/$PROJECT_ID/central-researcher-api:latest \
  --platform managed \
  --region asia-northeast1
```

### エラー2: CORS エラー

**症状**: ブラウザのコンソールに「CORS policy」エラー

**解決策**: Flask appでCORSが正しく設定されているか確認してください。`app.py`には既に`Flask-CORS`が設定されています。

### エラー3: Cloud Run のタイムアウト

**症状**: 分析実行中に504エラー

**解決策**:
```bash
# タイムアウトを延長（最大3600秒 = 1時間）
gcloud run services update central-researcher-api \
  --timeout 3600 \
  --region asia-northeast1
```

### エラー4: メモリ不足

**症状**: Cloud Runが「Memory limit exceeded」エラー

**解決策**:
```bash
# メモリを増量（最大8Gi）
gcloud run services update central-researcher-api \
  --memory 4Gi \
  --region asia-northeast1
```

---

## 📊 ログの確認

### Cloud Runのログ

```bash
# リアルタイムでログを確認
gcloud run services logs tail central-researcher-api \
  --region asia-northeast1

# 特定期間のログを確認
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=central-researcher-api" \
  --limit 50 \
  --format json
```

### Firebase Hostingのログ

Firebase Consoleから確認:
1. https://console.firebase.google.com/ にアクセス
2. プロジェクトを選択
3. Hosting → ダッシュボード

---

## 💰 コスト管理

### Cloud Run の料金

- **無料枠**: 毎月200万リクエストまで無料
- **料金**: リクエスト数、CPU時間、メモリ使用量に基づく

### 料金の確認

```bash
# 現在の使用状況を確認（GCP Console）
gcloud billing accounts list
```

### コスト削減のヒント

1. **最小インスタンス数を0に設定** (既に設定済み)
2. **不要な時はサービスを削除**:
   ```bash
   gcloud run services delete central-researcher-api --region asia-northeast1
   ```
3. **メモリとCPUを必要最小限に設定**

---

## 🔄 更新とメンテナンス

### バックエンドの更新

```bash
# コードを修正後
docker build -t gcr.io/$PROJECT_ID/central-researcher-api:latest .
docker push gcr.io/$PROJECT_ID/central-researcher-api:latest
gcloud run deploy central-researcher-api \
  --image gcr.io/$PROJECT_ID/central-researcher-api:latest \
  --platform managed \
  --region asia-northeast1
```

### フロントエンドの更新

```bash
# public/ 内のHTMLファイルを修正後
firebase deploy --only hosting
```

---

## 🔐 セキュリティ設定（オプション）

### 認証の追加

特定のユーザーのみにアクセスを制限したい場合:

1. **Firebase Authentication を有効化**
2. **Hosting のセキュリティルールを設定**
3. **Cloud Run の認証を有効化**

詳細は Firebase Authentication のドキュメントを参照してください。

---

## 📚 参考リンク

- [Firebase Hosting ドキュメント](https://firebase.google.com/docs/hosting)
- [Cloud Run ドキュメント](https://cloud.google.com/run/docs)
- [Flask デプロイガイド](https://flask.palletsprojects.com/en/2.3.x/deploying/)

---

## ✅ チェックリスト

デプロイ前に以下を確認してください:

- [ ] Firebase CLI がインストール済み
- [ ] Google Cloud SDK (gcloud) がインストール済み
- [ ] Docker がインストール済み
- [ ] `.firebaserc` にプロジェクトIDを設定
- [ ] `article_search/vectors/` ディレクトリにベクトルデータが存在
- [ ] Cloud Run APIs が有効化済み
- [ ] `public/index.html` のAPI URLを更新

デプロイ後:

- [ ] Firebase Hosting URLにアクセス可能
- [ ] API Health Check (`https://your-cloud-run-url/api/health`) が成功
- [ ] テストクエリで分析が実行できる
- [ ] 結果ページが正しく表示される

---

## 🎯 次のステップ

1. **カスタムドメインの設定** (オプション)
   ```bash
   firebase hosting:channel:deploy production
   ```

2. **パフォーマンス最適化**
   - Cloud CDNの有効化
   - キャッシュ戦略の調整

3. **モニタリングの設定**
   - Cloud Monitoring でアラートを設定
   - ダッシュボードの作成

---

## 📞 サポート

問題が発生した場合は、以下を確認してください:

1. Cloud Run ログ
2. Firebase Hosting ログ
3. ブラウザのコンソールログ

それでも解決しない場合は、エラーメッセージを含めてサポートにお問い合わせください。
