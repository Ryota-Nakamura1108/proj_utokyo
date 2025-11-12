# Keyword Extractor for Researchers

研究者の学術的専門性を表すキーワードを、OpenAlexの論文データから自動的に抽出するシステム。埋め込みベクトル化、クラスタリング、LLMベースの解析を組み合わせて、研究者の主要な研究テーマとキーワードを抽出し、包括的な英語サマリーを生成します。

## 目次

- [主な機能](#主な機能)
- [インストール](#インストール)
- [環境設定](#環境設定)
- [使用方法](#使用方法)
  - [CLI使用方法](#cli使用方法)
  - [Python API使用方法](#python-api使用方法)
- [パラメータ詳細](#パラメータ詳細)
- [出力形式](#出力形式)
- [技術仕様](#技術仕様)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [実装詳細](#実装詳細)
- [使用例](#使用例)
- [トラブルシューティング](#トラブルシューティング)

## 主な機能

1. **OpenAlexデータ取得**: 研究者名またはOpenAlex IDから論文データを自動取得
2. **論文フィルタリング**: 期間、著者位置（第一著者/最終著者）、引用数で論文を絞り込み
3. **埋め込み生成**: OpenAI `text-embedding-3-small` で論文を1536次元ベクトル化
4. **クラスタリング**: K-Meansによる論文の自動クラスタリング（研究テーマごとにグループ化）
5. **代表論文抽出**: 各クラスタから中心性と引用数を考慮した代表論文を抽出
6. **キーワード抽出**: LLM（GPT-4o-mini）によるコンテキスト理解型キーワード生成
7. **研究者サマリー生成**: キーワードと代表論文を基にした包括的な英語サマリー（250-350語）
8. **柔軟な出力形式**: CLI表示、Excel、JSONでの出力に対応

## インストール

### 前提条件

- Python 3.10以上
- UV パッケージマネージャー
- OpenAI APIキー
- OpenAlex APIアクセス用メールアドレス（オプション、Polite Poolアクセス用）

### 依存関係のインストール

```bash
# プロジェクトディレクトリで実行
cd /path/to/central_researcher
uv sync
```

自動的に以下の依存関係がインストールされます:
- `openai` - OpenAI API クライアント
- `httpx` - 非同期HTTPクライアント（OpenAlex API用）
- `scikit-learn` - クラスタリング処理
- `numpy` - 数値計算
- `python-dotenv` - 環境変数管理
- `pandas` - データ処理
- `openpyxl` - Excel出力

## 環境設定

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定してください：

```bash
# .env ファイル

# 必須: OpenAI APIキー
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# オプション: OpenAlex Polite Pool用メールアドレス
# 設定するとAPI制限が緩和されます
OPENALEX_EMAIL=your.email@example.com
```

### APIキーの取得

1. **OpenAI APIキー**: https://platform.openai.com/api-keys で取得
2. **OpenAlex Email**: OpenAlexのPolite Poolを利用する場合、自分のメールアドレスを設定（無料、登録不要）

## 使用方法

### CLI使用方法

#### 基本的な使い方

```bash
# 研究者名で検索
uv run keyword_extractor "Geoffrey Hinton"

# OpenAlex IDで検索
uv run keyword_extractor A1234567890
```

#### オプション付き実行

```bash
# Excelファイルにエクスポート
uv run keyword_extractor "Yoshua Bengio" --export results/bengio_keywords.xlsx

# JSONファイルにエクスポート
uv run keyword_extractor "Andrew Ng" --export results/ng_keywords.json

# 最大キーワード数を指定
uv run keyword_extractor "Yann LeCun" --max-keywords 15

# 過去5年間のデータのみ分析
uv run keyword_extractor "Fei-Fei Li" --years-back 5

# 最小引用数10以上の論文のみ使用
uv run keyword_extractor "Demis Hassabis" --min-citations 10

# クラスタ情報も含めて表示
uv run keyword_extractor A1234567890 --include-clusters

# 詳細ログを表示
uv run keyword_extractor "Judea Pearl" --verbose
```

#### すべてのオプション

```bash
uv run keyword_extractor --help
```

**オプション一覧:**

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|--------|------|----------|
| `--export PATH` | - | 結果をExcel/JSONファイルにエクスポート | なし |
| `--max-keywords N` | - | 抽出する最大キーワード数 | 10 |
| `--years-back N` | - | 遡る年数 | 10 |
| `--min-citations N` | - | 最小引用数フィルタ | 0 |
| `--include-clusters` | - | クラスタ情報を出力に含める | False |
| `--verbose` | `-v` | 詳細ログを表示 | False |

### Python API使用方法

#### 基本的な使い方

```python
from common_module.keyword_extractor import extract_keywords

# 研究者名で検索
result = extract_keywords("Geoffrey Hinton")

# OpenAlex IDで検索
result = extract_keywords("A1234567890")

# 結果を表示
print(f"Researcher: {result['researcher_name']}")
print(f"Keywords: {[kw['keyword'] for kw in result['keywords']]}")
print(f"\nSummary:\n{result['summary']}")
```

#### パラメータ付き実行

```python
from common_module.keyword_extractor import extract_keywords

# 詳細な設定で実行
result = extract_keywords(
    identifier="Yoshua Bengio",
    years_back=5,              # 過去5年間
    max_keywords=15,           # 最大15キーワード
    min_citations=10,          # 最小引用数10
    include_clusters=True,     # クラスタ情報を含める
    openai_api_key="sk-xxx",   # APIキーを直接指定（オプション）
    openalex_email="me@example.com"  # メールアドレスを直接指定（オプション）
)

# 統計情報の取得
stats = result['statistics']
print(f"Total papers analyzed: {stats['total_papers']}")
print(f"Total citations: {stats['total_citations']}")
print(f"Average citations: {stats['average_citations']:.1f}")

# キーワードの取得
for i, kw in enumerate(result['keywords'], 1):
    print(f"{i}. {kw['keyword']} (score: {kw['relevance_score']:.3f})")

# サマリーの取得
print(f"\nResearcher Summary:\n{result['summary']}")

# クラスタ情報の取得（include_clusters=True の場合）
if result.get('clusters'):
    for cluster in result['clusters']:
        print(f"\nCluster {cluster['cluster_id'] + 1}: {cluster['theme']}")
        print(f"  Size: {cluster['size']} papers")
        for paper in cluster['representative_papers'][:2]:
            print(f"  - {paper['title'][:60]}... ({paper['year']})")
```

#### 特定の関数を使用

```python
from common_module.keyword_extractor import (
    extract_keywords_by_name,
    extract_keywords_by_id
)

# 研究者名で検索
result = extract_keywords_by_name(
    researcher_name="Andrew Ng",
    years_back=10,
    max_keywords=10
)

# OpenAlex IDで検索
result = extract_keywords_by_id(
    openalex_id="A1234567890",
    min_citations=5,
    include_clusters=True
)
```

#### 非同期実行

```python
import asyncio
from common_module.keyword_extractor.core import KeywordExtractor

async def analyze_researcher():
    # KeywordExtractorインスタンスを作成
    extractor = KeywordExtractor(
        openai_api_key="sk-xxx",
        openalex_email="me@example.com"
    )

    # 非同期で実行
    result = await extractor.extract_by_name(
        researcher_name="Yann LeCun",
        years_back=10,
        max_keywords=10,
        min_citations=5,
        include_clusters=False
    )

    return result

# 実行
result = asyncio.run(analyze_researcher())
```

#### エクスポート機能

```python
from common_module.keyword_extractor import extract_keywords
from common_module.keyword_extractor.export import export_results

# キーワード抽出
result = extract_keywords("Geoffrey Hinton", max_keywords=15)

# Excelにエクスポート
export_results(result, "output/hinton_keywords.xlsx")

# JSONにエクスポート
export_results(result, "output/hinton_keywords.json")
```

## パラメータ詳細

### `identifier` / `researcher_name` / `openalex_id`

研究者の識別子。以下の形式をサポート:

- **研究者名**: `"Geoffrey Hinton"`, `"Yoshua Bengio"` など
- **OpenAlex ID**: `"A1234567890"` 形式または完全URL `"https://openalex.org/A1234567890"`

### `years_back` (デフォルト: 10)

分析対象とする過去の年数。

- `years_back=5`: 過去5年間の論文のみ分析
- `years_back=10`: 過去10年間の論文を分析（デフォルト）
- `years_back=20`: 過去20年間の論文を分析

### `max_keywords` (デフォルト: 10)

抽出する最大キーワード数。

- 推奨範囲: 5〜20
- 少なすぎると研究範囲を十分にカバーできない可能性
- 多すぎるとノイズが増える可能性

### `min_citations` (デフォルト: 0)

論文を分析対象に含めるための最小引用数。

- `min_citations=0`: すべての論文を含める（デフォルト）
- `min_citations=10`: 10回以上引用された論文のみ
- 影響力の高い論文にフォーカスしたい場合に有効

### `include_clusters` (デフォルト: False)

クラスタ情報を出力に含めるかどうか。

- `True`: 各クラスタのテーマと代表論文を含める
- `False`: キーワードと統計情報のみ（デフォルト）
- 研究の詳細な分類を見たい場合は `True` に設定

## 出力形式

### 返り値の構造

`extract_keywords()` 関数は以下の構造の辞書を返します:

```python
{
    "researcher_id": "A1234567890",           # OpenAlex ID
    "researcher_name": "Geoffrey Hinton",     # 研究者名
    "analysis_period": {                      # 分析期間
        "start_year": 2014,
        "end_year": 2024
    },
    "statistics": {                           # 統計情報
        "total_papers": 45,                   # 分析対象論文数
        "papers_as_first_author": 12,         # 第一著者論文数
        "papers_as_last_author": 28,          # 最終著者論文数
        "total_citations": 15420,             # 総引用数
        "average_citations": 342.7,           # 平均引用数
        "year_range": {                       # 実際の出版年範囲
            "start": 2014,
            "end": 2024
        }
    },
    "keywords": [                             # 抽出されたキーワード
        {
            "keyword": "deep learning",
            "relevance_score": 0.95,          # 関連性スコア（0-1）
            "cluster_ids": [0, 1, 2],         # このキーワードが現れるクラスタ
            "frequency": 0.0,                 # 頻度（現在未使用）
            "trend": null                     # トレンド（現在未使用）
        },
        {
            "keyword": "neural networks",
            "relevance_score": 0.89,
            "cluster_ids": [0, 1],
            "frequency": 0.0,
            "trend": null
        }
        # ... 他のキーワード
    ],
    "summary": "Geoffrey Hinton's research profile demonstrates...",  # 研究者サマリー（英語、250-350語）
    "clusters": [                             # クラスタ情報（include_clusters=True の場合）
        {
            "cluster_id": 0,
            "theme": "Deep Learning Fundamentals",
            "size": 15,                       # クラスタ内の論文数
            "representative_papers": [        # 代表論文（最大3件）
                {
                    "title": "Deep Learning",
                    "year": 2015,
                    "citations": 5420,
                    "journal": "Nature"
                }
                # ... 他の代表論文
            ]
        }
        # ... 他のクラスタ
    ],
    "method": "embedding_clustering_llm",     # 使用した手法
    "status": "success",                      # ステータス
    "processing_time": 45.2,                  # 処理時間（秒）
    "timestamp": "2024-01-15T10:30:00"       # タイムスタンプ
}
```

### Excel出力形式

`--export output.xlsx` または `export_to_excel()` を使用すると、以下のシート構成でExcelファイルが生成されます:

#### Sheet 1: Keywords
| Rank | Keyword | Relevance Score |
|------|---------|-----------------|
| 1 | deep learning | 0.950 |
| 2 | neural networks | 0.890 |
| ... | ... | ... |

#### Sheet 2: Metadata
| Field | Value |
|-------|-------|
| Researcher Name | Geoffrey Hinton |
| Researcher ID | A1234567890 |
| Total Papers | 45 |
| Extraction Method | embedding_clustering_llm |
| Status | success |

#### Sheet 3: Summary
| Researcher Summary |
|--------------------|
| Geoffrey Hinton's research profile demonstrates... [完全なサマリーテキスト] |

### JSON出力形式

`--export output.json` または `export_to_json()` を使用すると、上記の返り値構造がそのままJSON形式で保存されます。

## 技術仕様

### システムアーキテクチャ

```
┌─────────────────────┐
│ 研究者ID/名前入力    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   OpenAlex API      │ ← 論文データ取得
│  (httpx async)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Paper Filter       │ ← 著者位置、年、引用数でフィルタ
│  (FilterConfig)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Embedding Generator │ ← OpenAI text-embedding-3-small
│  (OpenAI API)       │    1536次元ベクトル化
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Clustering Engine   │ ← K-Means クラスタリング
│  (scikit-learn)     │    自動クラスタ数決定
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Representative      │ ← 中心性+引用数による
│ Extractor           │    代表論文抽出
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Keyword Generator   │ ← GPT-4o-mini による
│  (LLM)              │    キーワード抽出
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Researcher          │ ← GPT-4o-mini による
│ Summarizer (LLM)    │    包括的サマリー生成
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   結果出力          │ ← CLI / Excel / JSON
└─────────────────────┘
```

### 使用API・モデル

| コンポーネント | API/ライブラリ | モデル/バージョン |
|---------------|---------------|------------------|
| 論文データ取得 | OpenAlex API | v1 (REST API) |
| 埋め込み生成 | OpenAI API | text-embedding-3-small (1536次元) |
| キーワード抽出 | OpenAI API | gpt-4o-mini |
| サマリー生成 | OpenAI API | gpt-4o-mini |
| クラスタリング | scikit-learn | KMeans |
| HTTPクライアント | httpx | async |

### フィルタリング詳細

#### デフォルトフィルタ

```python
FilterConfig(
    years_back=10,                         # 過去10年
    author_positions=["first", "last"],    # 第一著者または最終著者
    min_citations=0,                       # 最小引用数なし
    max_papers=200,                        # 最大200論文
    paper_types=["article", "review"]      # 論文とレビューのみ
)
```

#### フィルタリングロジック

1. **期間フィルタ**: `publication_year >= current_year - years_back`
2. **著者位置フィルタ**: `author_position in ['first', 'last']`
   - 第一著者: 主導的研究を示す
   - 最終著者: 研究室主宰者としての研究を示す
3. **引用数フィルタ**: `cited_by_count >= min_citations`
4. **論文タイプ**: 学術論文とレビュー論文のみ（プレプリント等を除外）

### 埋め込み生成詳細

#### 入力テキスト構成

各論文について、以下の情報を結合してテキストを構成:

```python
embedding_text = f"""
Title: {paper.title}
Abstract: {paper.abstract}
Topics: {', '.join(paper.topics)}
Journal: {paper.journal_name}
Year: {paper.publication_year}
"""
```

#### バッチ処理

- バッチサイズ: 20論文/回
- 非同期処理により高速化
- フォールバック: アブストラクトがない場合、タイトルのみで埋め込み生成

### クラスタリング詳細

#### K-Means設定

```python
KMeans(
    n_clusters=n_clusters,    # 自動決定
    random_state=42,          # 再現性確保
    n_init=10                 # 初期化回数
)
```

#### クラスタ数の決定ロジック

```python
def determine_n_clusters(n_papers: int) -> int:
    if n_papers < 10:
        return min(3, n_papers)
    elif n_papers < 30:
        return 5
    elif n_papers < 60:
        return 7
    else:
        return min(10, int(np.sqrt(n_papers)))
```

**例:**
- 論文5件 → 3クラスタ
- 論文20件 → 5クラスタ
- 論文50件 → 7クラスタ
- 論文100件 → 10クラスタ

### 代表論文抽出詳細

#### Weighted Selection戦略（デフォルト）

中心性と引用数を組み合わせたスコアで選択:

```python
# 中心性スコア（クラスタ重心への距離）
distance_scores = 1 - normalized_distances  # 近いほど高スコア

# 引用スコア（対数スケール）
citation_scores = log1p(cited_by_count) / max(log1p(cited_by_count))

# 重み付けスコア
weights = 0.6 * distance_scores + 0.4 * citation_scores
```

**重み配分:**
- 中心性: 60% - テーマの代表性を重視
- 引用数: 40% - 影響力を考慮

詳細は `common_module/keyword_extractor/representative_extractor.py:239-244` を参照。

#### Medoid戦略（オプション）

クラスタの幾何学的中心に最も近い論文を選択。

### キーワード抽出詳細

#### LLMプロンプト設計

キーワード生成プロンプトの詳細は `common_module/keyword_extractor/keyword_generator.py:111-156` を参照。

**パラメータ:**
- **モデル**: `gpt-4o-mini`
- **Temperature**: 0.3 (一貫性重視)
- **Max Tokens**: 1000
- **出力形式**: JSON配列

### 研究者サマリー生成詳細

#### LLMプロンプト設計

研究者サマリー生成の詳細は `common_module/keyword_extractor/researcher_summarizer.py:73-133` を参照。

**パラメータ:**
- **モデル**: `gpt-4o-mini`
- **Temperature**: 0.7 (適度な多様性)
- **Max Tokens**: 800
- **目標文字数**: 250-350語

**サマリー構成:**
1. Opening: 研究分野の概要
2. Main Themes: 2-3の主要研究テーマと論文への言及
3. Research Impact: 引用数と出版ベニューに基づく影響力
4. Research Evolution: クラスタ間の研究の変遷
5. Conclusion: 分野への貢献の総合評価

## 実装詳細

### プロジェクト構成

```
common_module/keyword_extractor/
├── __init__.py                      # パッケージエントリポイント
├── __main__.py                      # CLI エントリポイント
├── cli.py                           # コマンドライン インターフェース
├── config.py                        # 設定管理（.env読み込み）
├── core.py                          # メインパイプライン
├── models.py                        # データモデル定義
├── openalex_fetcher.py              # OpenAlex API クライアント
├── paper_filter.py                  # 論文フィルタリング
├── embedding_generator.py           # 埋め込み生成
├── clustering_engine.py             # クラスタリング処理
├── representative_extractor.py      # 代表論文抽出
├── keyword_generator.py             # キーワード抽出
├── researcher_summarizer.py         # 研究者サマリー生成
├── export.py                        # エクスポート機能
├── SPEC.md                          # 技術仕様書
└── README.md                        # このファイル
```

### 主要コンポーネント

#### 1. OpenAlexFetcher (`openalex_fetcher.py`)

OpenAlex APIからの非同期データ取得。

**主要メソッド:**
- `search_author_by_name(name)`: 著者名検索
- `get_author_by_id(author_id)`: IDで著者取得
- `fetch_author_papers(author_id, years_back)`: 論文データ取得

#### 2. PaperFilter (`paper_filter.py`)

論文のフィルタリングと統計生成。

#### 3. EmbeddingGenerator (`embedding_generator.py`)

OpenAI APIによる埋め込み生成（バッチ処理、非同期）。

#### 4. ClusteringEngine (`clustering_engine.py`)

K-Meansによるクラスタリング（自動クラスタ数決定）。

#### 5. RepresentativeExtractor (`representative_extractor.py`)

各クラスタからの代表論文抽出（Weighted/Medoid戦略）。

#### 6. KeywordGenerator (`keyword_generator.py`)

LLMによるキーワード抽出（JSON形式出力）。

#### 7. ResearcherSummarizer (`researcher_summarizer.py`)

研究者プロファイルの包括的サマリー生成（キーワード+論文統合分析）。

#### 8. Core Pipeline (`core.py`)

全パイプラインの統合（`core.py:177-284` 参照）。

## 使用例

### 例1: 基本的なキーワード抽出

```bash
uv run keyword_extractor "Geoffrey Hinton"
```

### 例2: クラスタ情報を含めた詳細分析

```bash
uv run keyword_extractor "Yoshua Bengio" --include-clusters --export bengio_analysis.xlsx
```

### 例3: 高引用論文に絞った分析

```bash
uv run keyword_extractor "Andrew Ng" --min-citations 50 --years-back 5
```

### 例4: Pythonスクリプトでの使用

```python
from common_module.keyword_extractor import extract_keywords
from common_module.keyword_extractor.export import export_results

# 複数の研究者を分析
researchers = ["Geoffrey Hinton", "Yoshua Bengio", "Yann LeCun"]

for researcher in researchers:
    print(f"Analyzing {researcher}...")

    result = extract_keywords(
        identifier=researcher,
        years_back=10,
        max_keywords=15,
        min_citations=10,
        include_clusters=True
    )

    # 結果を表示
    print(f"\n{researcher}:")
    print(f"  Total papers: {result['statistics']['total_papers']}")
    print(f"  Top keywords: {[kw['keyword'] for kw in result['keywords'][:5]]}")

    # Excelに保存
    filename = f"results/{researcher.replace(' ', '_')}_keywords.xlsx"
    export_results(result, filename)
    print(f"  Exported to: {filename}\n")
```

### 例5: 特定のOpenAlex IDリストを処理

```python
from common_module.keyword_extractor import extract_keywords_by_id
import json

# OpenAlex IDのリスト
author_ids = ["A1234567890", "A9876543210", "A5555555555"]

results = {}

for author_id in author_ids:
    result = extract_keywords_by_id(
        openalex_id=author_id,
        max_keywords=10,
        years_back=10
    )

    results[author_id] = {
        "name": result["researcher_name"],
        "keywords": [kw["keyword"] for kw in result["keywords"]],
        "summary": result["summary"],
        "total_papers": result["statistics"]["total_papers"]
    }

# JSONに保存
with open("batch_analysis.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
```

## トラブルシューティング

### エラー: `OpenAI API key is required`

**原因:** `.env` ファイルに `OPENAI_API_KEY` が設定されていない。

**解決方法:**
1. プロジェクトルートに `.env` ファイルを作成
2. `OPENAI_API_KEY=sk-xxxxxxxx` を追加

### エラー: `Author not found`

**原因:** 指定した研究者名またはIDが見つからない。

**解決方法:**
1. 研究者名のスペルを確認
2. OpenAlexで事前に検索: https://openalex.org/
3. OpenAlex IDを直接使用（より確実）

### エラー: `No papers found`

**原因:** フィルタ条件が厳しすぎる。

**解決方法:**
1. `--years-back` を増やす
2. `--min-citations` を下げる

### エラー: `pandas is required for Excel export`

**解決方法:**
```bash
uv add pandas openpyxl
```

### 警告: `Using fallback summary generation`

**原因:** LLMによるサマリー生成が失敗。

**解決方法:**
1. インターネット接続を確認
2. OpenAI APIの使用状況を確認
3. しばらく待ってから再試行

### 処理が遅い

**解決方法:**
1. `--min-citations` を設定して論文数を減らす
2. `--years-back` を短くする
3. Polite Pool用に `OPENALEX_EMAIL` を設定

## 参考リンク

- **OpenAlex API**: https://docs.openalex.org/
- **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
- **OpenAI Chat Completions**: https://platform.openai.com/docs/guides/chat
- **scikit-learn KMeans**: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
- **技術仕様書**: `SPEC.md`

---

**Version**: 0.1.0
**Last Updated**: 2025-01-15
