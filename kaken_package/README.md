# Kaken Info Scraper

このパッケージは、研究者名を指定してNRID（研究者リゾルバー）から科研費の助成金情報をスクレイピングし、JSONLファイルとして保存するCLIツールを提供します。

## セットアップ

1.  リポジトリをクローンするか、上記のファイル群をディレクトリに配置します。
2.  （推奨）仮想環境を作成して有効化します。
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
3.  `uv` を使用して依存関係をインストールし、パッケージを編集可能モード（-e）でインストールします。
    ```bash
    uv pip install -e .
    ```
    (または `pip install -e .`)
4.  `.env.example` をコピーして `.env` を作成し、必要に応じてAPIキーを設定します。
    ```bash
    cp .env.example .env
    ```

## 実行方法

`pyproject.toml` の `[project.scripts]` 設定により、`uv run` コマンドで `kaken_info` スクリプトを実行できます。

### 基本的な実行

```bash
uv run kaken_info "研究者名"
# 例: uv run kaken_info "山田 太郎"

uv run kaken_info "研究者名" --institution "所属機関名"
# 例: uv run kaken_info "山田 太郎" --institution "東京大学"