# Prepare Module - 研究者リスト取得ツール

東京大学の研究者リストをOpenAlex APIから取得・管理するモジュールです。

## 概要

このモジュールは以下の機能を提供します：

- OpenAlex APIを使用した研究者データの取得
- 機関IDベースの研究者検索
- 論文数・被引用数によるフィルタリング
- 自動キャッシュ機能（30日間有効）
- レート制限対応（10リクエスト/秒）

## ファイル構成

```
prepare/
├── __init__.py              # パッケージ初期化
├── config.py                # 設定管理（環境変数、ディレクトリパス）
├── openalex_service.py      # OpenAlex APIクライアント
├── researcher_manager.py    # 研究者リスト管理（メインモジュール）
├── fetch_researchers.py     # 実行用スクリプト
└── README.md               # このファイル
```

## 使い方

### 基本的な使用方法

```bash
# デフォルト設定で実行（最小論文数: 5）
python fetch_researchers.py

# 最小論文数を指定
python fetch_researchers.py --min-works 100

# キャッシュを無視して再取得
python fetch_researchers.py --force-refresh

# カスタム出力先を指定
python fetch_researchers.py --output custom_output.json
```

### コマンドラインオプション

| オプション | 説明 | デフォルト値 |
|----------|------|------------|
| `--institution-id` | OpenAlex機関ID | `I74801974` (東京大学) |
| `--min-works` | 最小論文数フィルタ | `5` |
| `--force-refresh` | キャッシュを無視して再取得 | `False` |
| `--cache-days` | キャッシュ有効期限（日数） | `30` |
| `--output` | カスタム出力先パス | キャッシュディレクトリ |

### Pythonコードから使用

```python
from prepare import ResearcherManager

# 基本的な使用（非同期）
import asyncio

async def main():
    manager = ResearcherManager()
    researchers = await manager.fetch_researchers(min_works=10)
    print(f"取得した研究者数: {len(researchers)}")

    for r in researchers[:5]:
        print(f"{r['name']}: {r['works_count']} works")

asyncio.run(main())
```

```python
# 同期版の使用
from prepare import ResearcherManager

researchers = ResearcherManager.fetch_researchers_sync(
    institution_id="I74801974",
    min_works=100,
    force_refresh=False
)

print(f"取得した研究者数: {len(researchers)}")
```

## 出力形式

研究者リストは以下の形式のJSONファイルとして保存されます：

```json
{
  "institution_id": "I74801974",
  "cached_at": "2025-11-05T13:23:40.733927",
  "min_works": 100,
  "total_researchers": 3750,
  "researchers": [
    {
      "id": "A5065735206",
      "name": "Nobutaka Hirokawa",
      "works_count": 4017,
      "cited_by_count": 55085,
      "institutions": [
        "The University of Tokyo",
        "Juntendo University"
      ]
    },
    ...
  ]
}
```

## キャッシュ

- デフォルトのキャッシュディレクトリ: `../output/researcher_list/`
- キャッシュファイル名: `{institution_id}_researchers.json`
- キャッシュ有効期限: 30日間（`--cache-days`で変更可能）
- キャッシュが有効な場合は自動的に使用されます
- `--force-refresh`オプションでキャッシュを無視できます

## 環境変数

`.env`ファイルに以下の環境変数を設定できます：

```bash
# OpenAlex API（推奨）
OPENALEX_EMAIL=your-email@example.com

# デフォルト設定（オプション）
OPENALEX_INSTITUTION_ID=https://openalex.org/I74801974
MIN_WORKS=5
```

## 使用例

### 例1: 東京大学の全研究者を取得

```bash
python fetch_researchers.py --min-works 1
```

### 例2: 論文数100以上の研究者のみ取得

```bash
python fetch_researchers.py --min-works 100
```

### 例3: 他の大学の研究者を取得

```bash
# 京都大学（例: I136199984）
python fetch_researchers.py --institution-id I136199984 --min-works 50
```

### 例4: 定期的な更新（キャッシュ無視）

```bash
python fetch_researchers.py --force-refresh --min-works 10
```

## API制限

- **レート制限**: 10リクエスト/秒
- **日次制限**: 100,000リクエスト/日
- **Polite Pool**: メールアドレスを設定することで優先アクセス可能

## トラブルシューティング

### エラー: "ImportError: attempted relative import"

→ `fetch_researchers.py`を使用して実行してください。

### エラー: "Daily rate limit exceeded"

→ 1日あたり100,000リクエストの制限に達しました。翌日まで待つか、別のメールアドレスを使用してください。

### キャッシュが更新されない

→ `--force-refresh`オプションを使用してください。

## 開発者向け情報

### テスト実行

```bash
# ヘルプメッセージの確認
python fetch_researchers.py --help

# 小規模テスト（論文数1000以上の研究者のみ）
python fetch_researchers.py --min-works 1000

# インポートテスト
python -c "from prepare import ResearcherManager; print('OK')"
```

### モジュール構造

- `config.py`: 設定管理、環境変数読み込み
- `openalex_service.py`: OpenAlex APIクライアント（非同期版・同期版）
- `researcher_manager.py`: 研究者リスト管理、キャッシュ機能
- `fetch_researchers.py`: CLIエントリーポイント

## 関連リンク

- [OpenAlex API Documentation](https://docs.openalex.org/)
- [OpenAlex Institution Search](https://openalex.org/institutions)
- [東京大学のOpenAlexページ](https://openalex.org/institutions/I74801974)

## ライセンス

このプロジェクトの一部として提供されます。
