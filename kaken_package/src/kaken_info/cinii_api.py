# src/kaken_info/cinii_api.py
import requests
import pandas as pd
import os

def fetch_cinii_projects(researcher_number: int):
    """
    CiNii APIから指定された研究者番号のプロジェクトリストを取得します。
    環境変数 'CINII_APPID' からAPPIDを読み込みます。
    """
    # 環境変数から読み込み、なければNotebookのハードコード値を使用
    appid = os.getenv("CINII_APPID", '35CMT0qqSLVN3IP2a1ID')
    if appid == '35CMT0qqSLVN3IP2a1ID':
        print("警告: 環境変数 'CINII_APPID' が未設定。フォールバックIDを使用します。")

    url = f"https://ci.nii.ac.jp/nrid/10000{researcher_number}.json?appid={appid}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        name_entries = data.get('foaf:Person', [{}])[0].get('foaf:name', [])
        name_ja = next((n['@value'] for n in name_entries if n.get('@language') == 'ja'), None)
        affiliation_data = data.get('career', [{}])
        affiliation = None
        if affiliation_data and affiliation_data[0]:
             affiliation = (affiliation_data[0]
                             .get('institution', {})
                             .get('notation', [{}])[0]
                             .get('@value'))

        project_records = []
        for proj in data.get('project', []):
            kaken_id = next((i['@value'] for i in proj.get('projectIdentifier', []) if i['@type']=='KAKEN'), None)
            title_ja = next((t['@value'] for t in proj.get('notation',[]) if t.get('@language')=='ja'), None)
            title_en = next((t['@value'] for t in proj.get('notation',[]) if t.get('@language')=='en'), None)
            project_records.append({
                '研究者名': name_ja,
                '所属機関': affiliation,
                '科研費ID': kaken_id,
                '課題名（和）': title_ja,
                '課題名（英）': title_en,
                '役割': proj.get('role')
            })

        return pd.DataFrame(project_records)

    except requests.exceptions.RequestException as e:
        print(f"APIリクエストに失敗しました: {e}")
        return pd.DataFrame()