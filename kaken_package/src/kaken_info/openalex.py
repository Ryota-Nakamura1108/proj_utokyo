# src/kaken_info/openalex.py
import pandas as pd
import requests
import os
from pyalex import Authors
from dotenv import load_dotenv

def setup_openalex_email():
    """環境変数 'OPENALEX_EMAIL' からメールアドレスを設定します。"""
    load_dotenv()
    # 環境変数から読み込み、なければNotebookのハードコード値を使用
    email = os.getenv("OPENALEX_EMAIL", "nryota161108@gmail.com")
    if email == "nryota161108@gmail.com":
        print("警告: 環境変数 'OPENALEX_EMAIL' が未設定。フォールバックEmailを使用します。")
    Authors.email = email
    return email

def _fetch_from_api(parm: dict[str, str], email: str) -> list[dict]:
    papers = []
    cursor = "*"
    headers = {"User-Agent": f"mailto:{email}"} # pyalexに倣いmailtoを使用
    
    while True:
        try:
            params = parm.copy()
            if cursor: params["cursor"] = cursor

            response = requests.get("https://api.openalex.org/works", params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                filtered_results = [
                    work for work in data.get("results", [])
                    if work.get("abstract_inverted_index")
                ]
                papers.extend(filtered_results)

                if len(papers) >= 5: # 5件見つかったら終了
                    papers = papers[:5]
                    break
                
                cursor = data["meta"].get("next_cursor", None)
            else:
                print(f"Request failed: {response.status_code} {response.text}")
                break
        except Exception as e:
            print(f"Request failed with error: {e}")
            break
        if not cursor:
            break
    return papers

def reconstruct_abstract(inverted_index):
    if not inverted_index: return ""
    try:
        abstract_length = max(max(indices) for indices in inverted_index.values()) + 1
        abstract = [''] * abstract_length
        for word, indices in inverted_index.items():
            for idx in indices: abstract[idx] = word
        reconstructed = ' '.join([word for word in abstract if word != ''])
        return reconstructed.replace("Abstract", "").strip()
    except ValueError:
        return ""

def fetch_works_for_authors(id_list: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    email = setup_openalex_email()
    all_titles, all_abstracts = [], []

    for author_id in id_list:
        author_id_url = f"httpsor://openalex.org/{author_id}"
        parm = {
            "filter": f"author.id:{author_id_url}",
            "per_page": 200,
            "sort": "publication_year:desc,cited_by_count:desc"
        }
        works = _fetch_from_api(parm, email)
        author_titles = [work.get("display_name", "") for work in works]
        author_abstracts = [reconstruct_abstract(work.get("abstract_inverted_index", {})) for work in works]
        
        # 5件に満たない場合 None で埋める
        while len(author_titles) < 5:
            author_titles.append(None)
            author_abstracts.append(None)

        all_titles.append(author_titles)
        all_abstracts.append(author_abstracts)

    title_df = pd.DataFrame(all_titles).transpose()
    abstract_df = pd.DataFrame(all_abstracts).transpose()
    title_df.columns = id_list
    abstract_df.columns = id_list
    return title_df, abstract_df