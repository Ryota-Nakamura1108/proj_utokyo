# src/kaken_info/rag_llm.py
import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Literal, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# --- Notebook セル[24] (LLMクラス定義) ---
class OpenAIBase:
    def __init__(self, model: str):
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("APIキーが見つかりません。環境変数 'OPENAI_API_KEY' を設定してください。")
        self.client = OpenAI(api_key=self.api_key)
        self.usage_in = 0
        self.usage_out = 0

    def create_openai_query(self, query: str):
        return [{"role": "user", "content": query}]

    def get_response(
        self, query: str, temperature=0, max_tokens=15000,
        top_p=0, frequency_penalty=0, presence_penalty=0
    ) -> Tuple[str, int, int]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.create_openai_query(query=query),
            temperature=temperature, max_tokens=max_tokens, top_p=top_p,
            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty
        )
        usage_in = response.usage.prompt_tokens
        usage_out = response.usage.completion_tokens
        return response.choices[0].message.content, usage_in, usage_out

class LLM:
    GPT_3_OPENAI = "gpt-3.5-turbo"
    GPT_4O_OPENAI = "gpt-4o"
    GPT_4O_MINI_OPENAI = "gpt-4o-mini"
    MODEL_MAP = {
        "openai": {
            "gpt3": GPT_3_OPENAI, "gpt4o": GPT_4O_OPENAI, "gpt-4o-mini": GPT_4O_MINI_OPENAI, "gpt4o-mini": GPT_4O_MINI_OPENAI
        },
    }

    def __init__(self, base: Literal["openai"], use_model: Literal["gpt3", "gpt4o", "gpt-4o-mini", "gpt4o-mini"]):
        self.base = base
        self.use_model = use_model
        self.model = self.choose_model()

    def choose_model(self):
        # (以下、Notebook記載の通り)
        try:
            if self.base == "openai":
                return OpenAIBase(self.MODEL_MAP[self.base][self.use_model])
            else:
                raise ValueError(f"Invalid base: {self.base}")
        except KeyError:
            raise ValueError(f"Invalid model: {self.use_model}")

    def get_response(self, prompt: str, temperature=0, max_tokens=5000, top_p=1) -> Tuple[str, int, int]:
        response, in_usage, out_usage = self.model.get_response(
            query=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p
        )
        print("=======", self.base, self.use_model, "RESULT", "=======")
        print(response)
        print("=======", "\n")
        # (以下、Notebook記載の通り)
        found_responses = re.findall(r"[\{\[].*[\}\]]", response, re.DOTALL)
        if not found_responses:
            print("No valid response found.")
            return "", in_usage, out_usage
        response: str = found_responses[0]
        response: str = re.sub(r",\n\s*\}", "\n}", response)
        response: str = re.sub(r"\\", "", response)
        return response, in_usage, out_usage

# --- Notebook セル[26] (RAGロジック) ---
def parse_amount(s: str) -> float:
    if not isinstance(s, str): return np.nan
    m = re.search(r'([\d,]+)千円', s)
    return float(m.group(1).replace(',', '')) if m else np.nan

def load_and_preprocess_jsonl(jsonl_path: Path) -> pd.DataFrame:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"{jsonl_path} が見つかりません。")
    records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines()]
    df = pd.DataFrame(records)
    
    df['year'] = (
        df['研究期間 (年度)'].astype(str)
          .str.extract(r'(\d{4})').iloc[:, 0]
          .astype(float).fillna(0).astype(int)
    )
    df['amount'] = df['配分額*注記'].apply(parse_amount)
    return df

def run_rag_pipeline(jsonl_path_str: str, rag_query: str):
    """
    (注: この関数はベクトル検索部分(top5の定義)が
     元のNotebookから欠落しているため、動作しません)
    """
    print(f"RAGパイプラインを '{jsonl_path_str}' で開始します...")
    df = load_and_preprocess_jsonl(Path(jsonl_path_str))
    
    # --- ここにベクトル検索ロジック (top5の生成) が必要です ---
    # (例: SentenceTransformerでdfの概要をエンコードし、
    #  Faissでrag_queryとコサイン類似度検索を行う)
    
    print("警告: RAGのベクトル検索部分が未実装です (元のNotebookに欠落)。")
    # top5 = [] # 仮定義
    
    # --- 元のNotebookのセル[26]の 'top5' を使うコード (現在は実行不可) ---
    # context = ""
    # for rank, (sc, doc) in enumerate(top5, 1):
    #     ... (context構築) ...
    
    # print("■ RAGで取得した引用・参照元の論文情報一覧 ■")
    # ...
    
    # llm = LLM(base="openai", use_model="gpt4o")
    # prompt = create_rag_prompt(context, rag_query)
    # resp, _, _ = llm.get_response(prompt)