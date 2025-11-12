# Central Researcher Analysis - Enhanced Version

このプロジェクトは、研究者の中心性分析に加えて、各研究者の詳細情報（KAKENHI助成金情報、研究キーワード、研究サマリー）を個別のページで表示する拡張版システムです。

## 概要

このシステムは以下の3つのパッケージを統合しています：

1. **researcher_search**: 研究者の中心性分析とランキング
2. **kaken_package**: KAKENHI（科研費）助成金情報の取得
3. **keyword_extractor**: 研究キーワードと研究者サマリーの抽出

## 機能

### 1. メインHTMLレポート
- 研究者ランキングテーブル
- "Researcher Summary"、"KAKENHI Grants Info"、"Keywords"の3つのカラムを削除
- 代わりに"Researcher Page"というカラムを追加し、各研究者の個人ページへのリンクを表示
- ネットワーク可視化
- 研究トピック分析
- タイムライン分析

### 2. 個人研究者ページ
各研究者について、以下の情報を整理して表示：
- **基本統計情報**: CRS Score、H-Index、Papers、Leadership Rate
- **研究サマリー**: AI生成による研究内容の要約
- **研究キーワード**: 主要な研究キーワードのリスト
- **KAKENHI助成金情報**: 科研費の詳細情報（研究期間、配分額、研究概要など）

## ファイル構成

```
central_researcher/
├── main.py                          # メインスクリプト
├── custom_html_generator.py         # カスタムHTMLレポートジェネレーター
├── README_main.md                   # このファイル
├── researcher_search/               # 研究者検索・分析パッケージ
├── kaken_package/                   # 科研費情報取得パッケージ
├── keyword_extractor/               # キーワード抽出パッケージ
└── central_researcher_output_csv/   # 出力ディレクトリ
    ├── csv_analysis_report_The_University_of_Tokyo.html  # メインレポート
    ├── csv_central_researchers_The_University_of_Tokyo.csv
    └── researcher_pages/            # 個人研究者ページ
        ├── A1234567890.html
        ├── A0987654321.html
        └── ...
```

## 使い方

### 前提条件

1. Python 3.8以上
2. 必要なパッケージのインストール：
   ```bash
   pip install pandas numpy networkx asyncio openai selenium webdriver-manager beautifulsoup4
   ```

3. OpenAI APIキーの設定（keyword_extractorで使用）:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

### 実行方法

```bash
cd /Users/nakamuraryouta/Desktop/Python/Mlab/projUT/src/central_researcher
python main.py
```

### 入力データ

デフォルトでは、`researcher_search/tokyo_uni_works_random_1000.csv 15-53-47-766.csv`から研究論文データを読み込みます。

CSVファイルには以下のカラムが必要です：
- `work_id`: OpenAlex work ID
- `doi`: DOI

### 出力

実行後、以下のファイルが生成されます：

1. **central_researcher_output_csv/csv_analysis_report_The_University_of_Tokyo.html**
   - メインの分析レポート
   - ランキングテーブルに"Researcher Page"リンクが含まれる

2. **central_researcher_output_csv/researcher_pages/{author_id}.html**
   - 各研究者の個人ページ（トップ20研究者分）
   - KAKENHI情報、研究サマリー、キーワードを表示

3. **central_researcher_output_csv/csv_central_researchers_The_University_of_Tokyo.csv**
   - ランキングデータのCSVファイル

## カスタマイズ

### 分析対象の研究者数を変更

`main.py`の最後の部分で`top_n`パラメータを変更：

```python
asyncio.run(analyze_research_from_csv(
    seed_papers,
    filter_institution=target_institution,
    top_n=50  # 20から50に変更
))
```

### 対象機関を変更

`main.py`の`target_institution`変数を変更：

```python
target_institution = "Kyoto University"  # 例: 京都大学に変更
```

### 入力論文数を変更

`main.py`の以下の部分を変更：

```python
for index, row in df.iterrows():
    if index < 100:  # 50から100に変更
        # ...
```

## 技術的な詳細

### データ収集フロー

1. **研究者ランキング分析**:
   - CSVから論文データを読み込み
   - OpenAlexAPIを使って関連する研究者と論文を取得
   - ネットワーク分析により研究者の中心性を計算
   - CRS (Central Researcher Score)を算出してランキング

2. **追加データ収集**（トップN研究者について）:
   - **KAKENHI情報**: `get_research_field_data(researcher_name)`を使用してNRIDから科研費情報を取得
   - **キーワード・サマリー**: `extract_keywords_by_id()`または`extract_keywords_by_name()`を使用してOpenAlexとOpenAI APIで抽出

3. **HTMLレポート生成**:
   - CustomHTMLReportGenerator を使用してメインレポートを生成
   - 各研究者について個別のHTMLページを生成

### CRS (Central Researcher Score)

CRSは以下の3つの要素を組み合わせた複合指標：

1. **ネットワーク中心性 (70%)**:
   - PageRank (40%)
   - Degree Centrality (20%)
   - Betweenness Centrality (10%)

2. **引用影響力 (15%)**:
   - H-Index (100%)

3. **リーダーシップスコア (15%)**:
   - Leadership Rate (50%)

## トラブルシューティング

### エラー: "WebDriverの起動に失敗しました"

Chromeブラウザがインストールされていることを確認してください。または、`kaken_package/src/kaken_info/scraper.py`のWebDriver設定を変更してください。

### エラー: "OpenAI API key is required"

OpenAI APIキーを環境変数として設定してください：
```bash
export OPENAI_API_KEY="your-api-key-here"
```

または、`keyword_extractor/config.py`で直接設定してください。

### KAKENHI情報が取得できない

研究者名が正確でない可能性があります。NRIDのウェブサイトで研究者名を確認してください。

## 注意事項

1. **API制限**: OpenAlex APIとOpenAI APIには利用制限があります。大量のデータを処理する場合は、適切なレート制限を設定してください。

2. **処理時間**: トップ20研究者の処理には、ネットワーク速度とAPI応答時間に応じて10-30分程度かかる場合があります。

3. **データの正確性**: KAKENHI情報とキーワードは自動取得されるため、研究者名の表記揺れなどにより正確でない場合があります。

## ライセンス

このプロジェクトは研究・教育目的でのみ使用してください。

## 更新履歴

- **2025-10-19**: 初版リリース
  - researcher_search、kaken_package、keyword_extractorの統合
  - カスタムHTMLレポートジェネレーターの実装
  - 個人研究者ページの生成機能追加
