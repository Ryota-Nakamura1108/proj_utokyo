# src/kaken_info/scraper.py
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from typing import Optional

COLS_TO_KEEP = [
    '研究期間 (年度)',
    '配分額*注記',
    'キーワード',
    '研究開始時の研究の概要',
    '研究実績の概要',
    '今後の研究の推進方策',
    '研究成果の概要',
    '研究成果の学術的意義や社会的意義',
    '研究分野',
    '研究概要'
]

def setup_driver() -> Optional[webdriver.Chrome]:
    """Selenium WebDriverをヘッドレスモードでセットアップします。"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), 
            options=chrome_options
        )
        return driver
    except Exception as e:
        print(f"WebDriverの起動に失敗しました: {e}")
        print("Chromeがインストールされているか、ネットワーク接続が有効か確認してください。")
        return None

def get_research_field_data(name: str, institution: None) -> Optional[pd.DataFrame]:
    """
    指定した研究者名でNRIDサイトを検索し、
    KAKEN助成金詳細ページを巡回してDataFrameを返します。
    """
    driver = setup_driver()
    if driver is None:
        return None

    all_records = []
    
    try:
        print(f"Searching for: {name}")
        driver.get('https://nrid.nii.ac.jp/ja/index/')
        
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'text_kywd_0'))
        )
        search_box.clear()
        search_box.send_keys(name, Keys.RETURN)
    
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//div[@class='listitem xfolkentry']"))
        )
        results = driver.find_elements(By.XPATH, "//div[@class='listitem xfolkentry']")
        
        if len(results) == 0:
            print(f"'{name}' の検索結果が見つかりませんでした。")
            return None
        if len(results) > 1:
            print(f"'{name}' の検索結果が複数見つかりました。{institution}で検索対象を絞り込みを開始します。")
            if institution is None:
                print(f"institutionが不明のため、検索結果の最初の結果を使います。")

            #TODO: 所属インステで絞り込む機能を追加する
            result = list()

            driver.get("https://nrid.nii.ac.jp/ja/index/")
            
            toggle = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a[data-target='#research-advance']"))
            )
            toggle.click()

            WebDriverWait(driver, 10).until(
                lambda d: "show" in d.find_element(By.ID, "research-advance").get_attribute("class")
            )
            search_box = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, "qg"))
            )
            search_box.clear()
            search_box.send_keys(name)

            search_box = WebDriverWait(driver, 10).until(
                EC.visibility_of_element_located((By.ID, "qh"))
            )
            search_box.clear()
            search_box.send_keys(institution)

            search_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "searchbtn_"))
            )
            search_button.click()

            print("検索を実行しました。")

            # --- 結果のロードを待機 ---
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "listitem"))
            )
            print("検索結果ページに到達しました。")

            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, "//div[@class='listitem xfolkentry']"))
            )
            results = driver.find_elements(By.XPATH, "//div[@class='listitem xfolkentry']")

        profile_link = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//a[@class='link-page']"))
        )
        profile_url = profile_link.get_attribute('href')
        driver.get(profile_url)
        
        grant_links_elements = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located(
                (By.CSS_SELECTOR, 'div.col-sm-12 h4 a[href*="KAKENHI-PROJECT-"]')
            )
        )
        grant_urls = [a.get_attribute('href') for a in grant_links_elements]
        print(f"Found {len(grant_urls)} grant URLs.")
        
        for i, url in enumerate(grant_urls, 1):
            print(f"Processing URL {i}/{len(grant_urls)}: {url.split('/')[-2]}")
            driver.get(url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'table'))
            )
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            record = {'grant_url': url}
            for tr in soup.find_all('tr'):
                th = tr.find('th')
                td = tr.find('td')
                if not th or not td:
                    continue
                key = th.get_text(strip=True)
                value = td.get_text(separator=' ', strip=True)
                record[key] = value
            all_records.append(record)
            time.sleep(0.3)
        
        if not all_records:
            return pd.DataFrame()

        return pd.DataFrame(all_records)
        
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        return None
    finally:
        if driver:
            driver.quit()