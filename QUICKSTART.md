# 🚀 クイックスタートガイド

Firebase Hosting + Cloud Runへのデプロイを簡単に行うためのガイドです。

## 📋 必要なもの

- [ ] Firebase CLI (`npm install -g firebase-tools`)
- [ ] Google Cloud SDK (`gcloud`)
- [ ] Docker
- [ ] Firebaseプロジェクト

## ⚡ 3ステップでデプロイ

### ステップ1: プロジェクト設定

```bash
# Firebaseプロジェクトを設定
firebase use --add
# プロジェクトを選択して、エイリアス名を "default" にする

# Google Cloud プロジェクトを設定
gcloud config set project YOUR_PROJECT_ID

# 必要なAPIを有効化
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

### ステップ2: ベクトルデータの配置

```bash
# ベクトルデータを配置（既にある場合はスキップ）
mkdir -p article_search/vectors
# あなたのベクトルファイルを article_search/vectors/ にコピー
```

### ステップ3: デプロイ

```bash
# デプロイスクリプトを実行
./deploy.sh all
```

デプロイが完了すると、以下のURLが表示されます:
- **Cloud Run URL**: `https://central-researcher-api-xxxxx.a.run.app`
- **Firebase Hosting URL**: `https://your-project-id.web.app`

### ステップ4: API URLの更新

`public/index.html` を開いて、45行目付近の API_URL を更新:

```javascript
const API_URL = 'https://your-cloud-run-url';  // ← Cloud Run URLに変更
```

再度フロントエンドをデプロイ:
```bash
firebase deploy --only hosting
```

## ✅ 動作確認

1. ブラウザで Firebase Hosting URL にアクセス
2. クエリを入力: `quantum computing`
3. メールアドレスを入力
4. 「分析開始」をクリック
5. 結果が表示されることを確認

## 🔧 よくある問題

### Cloud Run URLがわからない

```bash
gcloud run services describe central-researcher-api \
  --platform managed \
  --region asia-northeast1 \
  --format 'value(status.url)'
```

### ベクトルディレクトリのエラー

ベクトルデータが `article_search/vectors/` に存在することを確認してから再デプロイ:

```bash
./deploy.sh backend
```

### CORS エラー

`app.py` で CORS が正しく設定されているか確認。既に `Flask-CORS` が設定済みのはずです。

## 📚 詳細情報

詳しいデプロイ手順やトラブルシューティングは `DEPLOYMENT_GUIDE.md` を参照してください。

## 🎯 個別デプロイ

```bash
# バックエンドのみ
./deploy.sh backend

# フロントエンドのみ
./deploy.sh frontend
```

## 💡 ヒント

- 初回デプロイには10-15分程度かかります
- Cloud Runの無料枠: 月間200万リクエストまで
- Firebase Hostingの無料枠: 月間10GBまで
