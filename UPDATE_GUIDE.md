# 🔄 更新・メンテナンスガイド

デプロイ後のベクトルデータ更新、機能追加、メンテナンス方法を説明します。

## 📦 ベクトルデータの更新

### 方法1: Cloud Storageのデータを直接更新（推奨）

**最も簡単な方法**です。Cloud Runを再デプロイする必要はありません。

#### ステップ1: 新しいベクトルデータを準備

```bash
# 新しいベクトルデータを article_search/vectors/ に配置
ls -la article_search/vectors/
```

#### ステップ2: Cloud Storageに同期

```bash
# プロジェクトIDを確認
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-vectors"

# 新しいデータをCloud Storageにアップロード（既存ファイルを上書き）
gsutil -m rsync -r -d article_search/vectors/ gs://${BUCKET_NAME}/vectors/

# 確認
gsutil ls gs://${BUCKET_NAME}/vectors/ | wc -l
```

#### ステップ3: Cloud Runインスタンスをリセット

```bash
# 既存のインスタンスをすべて削除（次回リクエスト時に新しいインスタンスが起動）
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --clear-vpc-connector \
  --no-traffic

# トラフィックを戻す
gcloud run services update-traffic central-researcher-api \
  --region asia-northeast1 \
  --to-latest
```

または、もっと簡単に：

```bash
# 一時的にmin-instancesを0に設定（自動的にインスタンスが停止）
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0

# 数分待つと、次回のリクエスト時に新しいデータがダウンロードされます
```

**注意**: 新しいインスタンスが起動するまで、初回リクエストに時間がかかります（約2-5分）。

---

### 方法2: ベクトルファイルを個別に追加・削除

```bash
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-vectors"

# 新しいファイルを追加
gsutil cp article_search/vectors/paper_embeddings_part_0048.h5 \
  gs://${BUCKET_NAME}/vectors/

# ファイルを削除
gsutil rm gs://${BUCKET_NAME}/vectors/paper_embeddings_part_0001.h5

# 確認
gsutil ls gs://${BUCKET_NAME}/vectors/
```

---

## 🛠️ コードの機能追加・修正

### バックエンド（Flask API）の更新

#### ステップ1: コードを修正

```bash
# app.py や他のPythonファイルを編集
vim app.py
```

#### ステップ2: ローカルでテスト（オプション）

```bash
# ローカルで動作確認
python app.py

# 別のターミナルでテスト
curl http://localhost:5000/api/health
```

#### ステップ3: 再デプロイ

```bash
# バックエンドのみ再デプロイ
./deploy.sh backend
```

**所要時間**: 約5-10分

---

### フロントエンド（HTML）の更新

#### ステップ1: HTMLファイルを修正

```bash
# public/index.html や public/results.html を編集
vim public/index.html
```

#### ステップ2: デプロイ

```bash
# フロントエンドのみデプロイ
./deploy.sh frontend
```

**所要時間**: 約1-2分

---

## 📊 新しいAPIエンドポイントの追加

### 例: 統計情報取得エンドポイントを追加

#### 1. app.py に新しいエンドポイントを追加

```python
@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """全体の統計情報を取得"""
    try:
        # ここに処理を記述
        stats = {
            'total_papers': 123456,
            'total_researchers': 5678,
            'last_updated': datetime.now().isoformat()
        }
        return jsonify({
            'status': 'success',
            'data': stats
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500
```

#### 2. 再デプロイ

```bash
./deploy.sh backend
```

#### 3. フロントエンドから呼び出し

```javascript
// public/index.html
const response = await fetch(`${API_URL}/api/statistics`);
const result = await response.json();
console.log(result.data);
```

---

## 🔐 環境変数の追加・変更

### Cloud Runの環境変数を更新

```bash
# 環境変数を追加
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --set-env-vars NEW_VAR=value

# 複数の環境変数を設定
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --set-env-vars VAR1=value1,VAR2=value2

# 環境変数を確認
gcloud run services describe central-researcher-api \
  --region asia-northeast1 \
  --format 'value(spec.template.spec.containers[0].env)'
```

### 現在設定されている環境変数

- `GCS_BUCKET_NAME`: Cloud Storageバケット名
- `DOWNLOAD_VECTORS_ON_STARTUP`: 起動時にベクトルをダウンロードするか（true/false）
- `PORT`: ポート番号（8080）
- `PYTHONUNBUFFERED`: ログ出力設定

---

## 🔄 定期的なメンテナンス

### 月次チェックリスト

- [ ] Cloud Runの料金を確認
- [ ] Cloud Storageの使用量を確認
- [ ] ログにエラーがないか確認
- [ ] ベクトルデータが最新か確認

### 料金の確認

```bash
# 現在の料金を確認（GCP Console推奨）
# https://console.cloud.google.com/billing

# Cloud Runの使用状況
gcloud run services describe central-researcher-api \
  --region asia-northeast1 \
  --format json

# Cloud Storageの使用量
gsutil du -s gs://${PROJECT_ID}-vectors/
```

### ログの確認

```bash
# 最新のログを表示
gcloud run services logs read central-researcher-api \
  --region asia-northeast1 \
  --limit 50

# エラーログのみ表示
gcloud run services logs read central-researcher-api \
  --region asia-northeast1 \
  --limit 100 | grep ERROR
```

---

## 📈 パフォーマンスチューニング

### メモリとCPUの調整

```bash
# より多くのメモリが必要な場合
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --memory 4Gi

# CPUを増やす
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --cpu 4

# タイムアウトを延長
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --timeout 900
```

### 最小インスタンス数の設定（コールドスタート対策）

```bash
# 常に1インスタンスを起動しておく（レスポンス高速化、ただし料金増加）
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 1

# コストを抑える場合は0に戻す
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0
```

---

## 🗑️ クリーンアップ（サービスの削除）

### 一時的に停止したい場合

```bash
# トラフィックを0にする（課金は継続）
gcloud run services update-traffic central-researcher-api \
  --region asia-northeast1 \
  --to-revisions=REVISION=0
```

### 完全に削除したい場合

```bash
# 1. Cloud Runサービスを削除
gcloud run services delete central-researcher-api \
  --region asia-northeast1

# 2. Dockerイメージを削除
gcloud container images delete gcr.io/$PROJECT_ID/central-researcher-api:latest

# 3. Cloud Storageバケットを削除（データも削除されます！）
gsutil rm -r gs://${PROJECT_ID}-vectors/

# 4. Firebase Hostingを無効化
firebase hosting:disable
```

---

## 🔄 ベクトルデータの完全な置き換え

古いデータをすべて削除して、新しいデータに置き換える場合：

```bash
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME="${PROJECT_ID}-vectors"

# 1. 既存のデータを削除
gsutil -m rm gs://${BUCKET_NAME}/vectors/**

# 2. 新しいデータをアップロード
gsutil -m cp -r article_search/vectors/* gs://${BUCKET_NAME}/vectors/

# 3. 確認
gsutil ls gs://${BUCKET_NAME}/vectors/ | wc -l

# 4. Cloud Runインスタンスをリセット
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0
```

---

## 🐛 トラブルシューティング

### ベクトルデータが更新されない

**原因**: Cloud Runインスタンスがキャッシュを使用している

**解決策**:
```bash
# すべてのインスタンスを停止
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0 \
  --max-instances 0

# 数秒待つ

# インスタンス数を元に戻す
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --min-instances 0 \
  --max-instances 10
```

### デプロイに失敗する

**原因**: Dockerビルドエラー

**解決策**:
```bash
# ローカルでビルドテスト
docker build -t test-image .

# エラーメッセージを確認
docker build -t test-image . 2>&1 | tee build.log
```

### API呼び出しがタイムアウトする

**原因**: 処理時間が長すぎる

**解決策**:
```bash
# タイムアウトを延長
gcloud run services update central-researcher-api \
  --region asia-northeast1 \
  --timeout 1800  # 30分
```

---

## 📞 よくある質問

### Q1: ベクトルデータを更新したら、すぐに反映されますか？

A: いいえ。Cloud Runの既存インスタンスは古いデータを使用し続けます。インスタンスをリセットする必要があります。

### Q2: 機能追加のたびにベクトルデータを再アップロードする必要がありますか？

A: いいえ。コード変更のみの場合、`./deploy.sh backend` で再デプロイするだけです。ベクトルデータはCloud Storageに保存されているため、再アップロード不要です。

### Q3: デプロイ中にサービスは停止しますか？

A: いいえ。Cloud Runは新しいバージョンを段階的にロールアウトするため、ダウンタイムはありません。

### Q4: ローカルで開発する場合、Cloud Storageは必要ですか？

A: いいえ。ローカルでは `article_search/vectors/` 内のデータを直接使用できます。環境変数 `GCS_BUCKET_NAME` を設定しなければ、ローカルモードで動作します。

---

## 📚 関連ドキュメント

- [Cloud Run ドキュメント](https://cloud.google.com/run/docs)
- [Cloud Storage ドキュメント](https://cloud.google.com/storage/docs)
- [Firebase Hosting ドキュメント](https://firebase.google.com/docs/hosting)

---

## ✅ 更新フローのまとめ

### ベクトルデータを更新する場合

1. Cloud Storageに新しいデータをアップロード
2. Cloud Runインスタンスをリセット
3. 完了！

### コードを修正する場合

1. コードを編集
2. `./deploy.sh backend` または `./deploy.sh frontend`
3. 完了！

### 両方を更新する場合

1. Cloud Storageにデータをアップロード
2. コードを編集
3. `./deploy.sh all`
4. 完了！
