# src/kaken_info/cli.py
import argparse
import os
import pandas as pd
from dotenv import load_dotenv
from .scraper import get_research_field_data, COLS_TO_KEEP

def main():
    # .envファイル読み込み（もしあれば OpenAIキー等）
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="研究者名で科研費情報をNRIDからスクレイピングし、JSONLファイルとして保存します。"
    )
    parser.add_argument(
        "name",
        type=str,
        help="検索する研究者名 (例: \"Issei Komuro\")"
    )
    parser.add_argument(
        "--institution", 
        type=str,
        default=None,
        help="所属機関名 (検索結果が複数ある場合の絞り込み用)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./RAG_output",
        help="出力ディレクトリ (デフォルト: ./RAG_output)"
    )
    args = parser.parse_args()

    researcher_name = args.name
    institution_name = args.institution
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"研究者 '{researcher_name}' の科研費情報を取得します...")
    if institution_name:
        print(f"所属機関 '{institution_name}' で絞り込みます。")
    df = get_research_field_data(researcher_name, institution=institution_name)


    if df is not None and not df.empty:
        existing_cols = [col for col in COLS_TO_KEEP if col in df.columns]
        missing_cols = [col for col in COLS_TO_KEEP if col not in df.columns]

        if missing_cols:
            print(f"警告: 以下の必須カラムが一部のデータに存在しませんでした: {missing_cols}")

        df_selected = df[existing_cols]

        safe_filename = "".join(c if c.isalnum() else "_" for c in researcher_name)
        output_path = f"{output_dir}/{safe_filename}.jsonl"

        df_selected.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False
        )
        print(f"データを {output_path} に保存しました。")
    elif df is not None and df.empty:
        print(f"'{researcher_name}' の助成金情報は見つかりませんでした、または詳細の取得に失敗しました。")
    else:
        print(f"'{researcher_name}' のデータの取得に失敗しました。")

if __name__ == "__main__":
    main()