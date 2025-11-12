# Central Researcher Flask API

このFlask APIは、論文検索と研究者分析機能をWeb APIとして提供します。

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements-api.txt
```

### 2. APIサーバーの起動

```bash
python app.py
```

デフォルトでは `http://localhost:5000` でサーバーが起動します。

## API エンドポイント

### 1. ヘルスチェック

**エンドポイント:** `GET /api/health`

**説明:** APIサーバーの状態と利用可能なモジュールを確認します。

**レスポンス例:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-09T10:00:00",
  "modules": {
    "PaperSearchEngine": true,
    "CentralResearcher": true,
    "KAKENHI": true,
    "KeywordExtractor": true
  }
}
```

**cURLコマンド:**
```bash
curl http://localhost:5000/api/health
```

---

### 2. 研究者分析 (メインエンドポイント)

**エンドポイント:** `POST /api/analyze`

**説明:** 指定されたクエリに基づいて論文検索と研究者分析を実行します。

**リクエストボディ:**
```json
{
  "query": "quantum computing",
  "email": "your_email@example.com",
  "filter_institution": "The University of Tokyo",
  "top_k": 100,
  "similarity_threshold": 0.55
}
```

**パラメータ:**
- `query` (必須): 検索クエリ文字列
- `email` (必須): OpenAlex API用のメールアドレス
- `filter_institution` (オプション): フィルタする機関名 (デフォルト: "The University of Tokyo")
- `top_k` (オプション): 取得する論文の最大数 (デフォルト: 100)
- `similarity_threshold` (オプション): 類似度の閾値 (デフォルト: 0.55)

**レスポンス例:**
```json
{
  "status": "success",
  "data": {
    "query": "quantum computing",
    "filter_institution": "The University of Tokyo",
    "total_papers": 85,
    "total_researchers": 42,
    "rankings": [
      {
        "rank": 1,
        "author_id": "A1234567890",
        "author_name": "John Doe",
        "crs_score": 0.8543,
        "h_index": 45,
        "papers_in_corpus": 12,
        "leadership_rate": 0.75,
        "institutions": ["The University of Tokyo"]
      }
    ],
    "researcher_data": {
      "A1234567890": {
        "author_id": "A1234567890",
        "author_name": "John Doe",
        "kakenhi_info": [...],
        "keywords": ["quantum computing", "quantum algorithms"],
        "summary": "Research summary...",
        "statistics": {}
      }
    },
    "network_statistics": {
      "total_nodes": 42,
      "total_edges": 156,
      "density": 0.087,
      "average_clustering": 0.543
    },
    "timestamp": "2025-11-09T10:00:00"
  }
}
```

**cURLコマンド:**
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "query": "quantum computing",
    "email": "your_email@example.com",
    "filter_institution": "The University of Tokyo"
  }'
```

**Pythonでの使用例:**
```python
import requests

url = "http://localhost:5000/api/analyze"
data = {
    "query": "quantum computing",
    "email": "your_email@example.com",
    "filter_institution": "The University of Tokyo",
    "top_k": 100,
    "similarity_threshold": 0.55
}

response = requests.post(url, json=data)
result = response.json()

if result['status'] == 'success':
    print(f"Found {result['data']['total_researchers']} researchers")
    for ranking in result['data']['rankings'][:5]:
        print(f"{ranking['rank']}. {ranking['author_name']} - CRS: {ranking['crs_score']}")
else:
    print(f"Error: {result['error']}")
```

---

### 3. 研究者詳細情報取得

**エンドポイント:** `GET /api/researchers/<researcher_id>`

**説明:** 特定の研究者の詳細情報(KAKENHI情報、キーワード等)を取得します。

**クエリパラメータ:**
- `author_name` (必須): 研究者名

**レスポンス例:**
```json
{
  "status": "success",
  "data": {
    "author_id": "A1234567890",
    "author_name": "John Doe",
    "kakenhi_info": [
      {
        "研究課題名": "量子コンピューティングの研究",
        "研究期間": "2020-2023",
        "配分額": "10000000"
      }
    ],
    "keywords": [
      {"keyword": "quantum computing", "score": 0.95},
      {"keyword": "quantum algorithms", "score": 0.87}
    ],
    "summary": "This researcher focuses on quantum computing...",
    "statistics": {
      "total_papers": 45,
      "total_citations": 1234
    }
  }
}
```

**cURLコマンド:**
```bash
curl "http://localhost:5000/api/researchers/A1234567890?author_name=John%20Doe"
```

**Pythonでの使用例:**
```python
import requests

researcher_id = "A1234567890"
author_name = "John Doe"

url = f"http://localhost:5000/api/researchers/{researcher_id}"
params = {"author_name": author_name}

response = requests.get(url, params=params)
result = response.json()

if result['status'] == 'success':
    data = result['data']
    print(f"Researcher: {data['author_name']}")
    print(f"Keywords: {', '.join([k['keyword'] for k in data['keywords']])}")
else:
    print(f"Error: {result['error']}")
```

---

## エラーレスポンス

エラーが発生した場合、以下の形式でレスポンスが返されます:

```json
{
  "status": "error",
  "error": "エラーメッセージ"
}
```

**HTTPステータスコード:**
- `200`: 成功
- `400`: リクエストが不正
- `404`: リソースが見つからない
- `500`: サーバーエラー

---

## 注意事項

1. **ベクトルディレクトリ**: `./article_search/vectors` ディレクトリが存在することを確認してください。存在しない場合、論文検索が機能しません。

2. **メールアドレス**: OpenAlex APIを使用するため、有効なメールアドレスを指定してください。

3. **処理時間**: 分析には数分かかる場合があります。タイムアウト設定に注意してください。

4. **依存モジュール**: 以下のモジュールが正しくインポートされている必要があります:
   - `article_search.core.article_search`
   - `researcher_search.core.central_researcher`
   - `kaken_info.scraper` (オプション)
   - `keyword_extractor.core` (オプション)

---

## 開発モード

デバッグモードで起動する場合:

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

本番環境では、gunicornやuWSGIなどのWSGIサーバーを使用することを推奨します:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## CORS設定

デフォルトですべてのオリジンからのリクエストを許可しています。本番環境では、セキュリティのため適切なCORS設定を行ってください:

```python
from flask_cors import CORS

CORS(app, resources={r"/api/*": {"origins": "https://yourdomain.com"}})
```

---

## ログ

APIサーバーは標準出力にログを出力します。ログレベルは `logging.INFO` に設定されています。

```python
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
```
