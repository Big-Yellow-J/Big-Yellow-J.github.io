from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import xml.etree.ElementTree as ET
import os
import requests
from datetime import datetime, timedelta
import concurrent.futures

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

SCOPES = ['https://www.googleapis.com/auth/indexing']
CREDENTIALS_FILE = './google_api.json'
SITE_URL = 'https://www.big-yellow-j.top/'
SITEMAP_LOCAL_PATH = '_site/static/xml/sitemap.xml'
SITEMAP_URL = 'https://www.big-yellow-j.top/static/xml/sitemap.xml'
DAYS_TO_FILTER = 50

credentials = service_account.Credentials.from_service_account_file(
    CREDENTIALS_FILE, scopes=SCOPES)
service = build('indexing', 'v3', credentials=credentials)

def get_sitemap_urls(local_path, online_url, days=None):
    content = None
    source = None

    # 尝试从本地文件获取sitemap内容
    if os.path.exists(local_path):
        try:
            with open(local_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                source = 'local file'
                if not content.strip():
                    print(f"Sitemap file {local_path} is empty.")
                    content = None
        except Exception as e:
            print(f"Error reading local sitemap {local_path}: {e}")
            content = None

    # 如果本地文件不可用，尝试从线上获取sitemap内容
    if content is None:
        try:
            response = requests.get(online_url, timeout=10)
            response.raise_for_status()
            content = response.text
            source = 'online URL'
        except requests.RequestException as e:
            print(f"Error fetching online sitemap {online_url}: {e}")
            return []

    # 检查是否包含 Liquid 模板或 YAML front matter
    if '---' in content[:100] or '{%' in content:
        print(f"Error: Sitemap contains Liquid template or YAML front matter. Jekyll did not render it correctly.")
        return []

    # 解析 XML 并提取 URL
    try:
        tree = ET.fromstring(content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []

        # 如果传入了 days 参数，限制返回的 URL 数量
        if days is not None:
            urls = [url_elem.find('ns:loc', namespace).text for url_elem in tree.findall('.//ns:url', namespace)][:days]
        else:
            # 如果没有传入 days 参数，返回所有的 URL
            urls = [url_elem.find('ns:loc', namespace).text for url_elem in tree.findall('.//ns:url', namespace)]

        return urls
    except ET.ParseError as e:
        print(f"Error parsing sitemap: {e}")
        print(f"Sitemap content preview:\n{content[:500]}")
        return []
    except Exception as e:
        print(f"Unexpected error while parsing sitemap: {e}")
        return []

def google_submit_url(url):
    body = {
        'url': url,
        'type': 'URL_UPDATED'
    }
    try:
        response = service.urlNotifications().publish(body=body).execute()
        print(f"✅ Google submitted: {url}\n")
        return True
    except HttpError as error:
        print(f"❌ Google error {url}: {error.status_code} {error.reason}\n")
        return False
    except Exception as e:
        print(f"❌ Google other error {url}: {e}\n")
        return False

def bing_submit_url(url):
    endpoint = "https://api.indexnow.org/indexnow"
    domain = url.split('/')[2]
    payload = {
        "host": domain,
        "key": 'd8eac6ca9eb040f28f6ed4ab5cd8d5ad',
        "keyLocation": 'https://www.big-yellow-j.top/Bing_indexnow.txt',
        "urlList": [url]
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code in [200, 202]:
            print(f"✅ Bing submitted: {url}")
            return True
        else:
            print(f"❌ Bing failed {url}: Status {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Bing exception {url}: {e}")
        return False

def submit_url(url):
    """同时提交到 Google 和 Bing"""
    try:
        google_submit_url(url)
        bing_submit_url(url)
    except Exception:
        pass

if __name__ == '__main__':
    urls = get_sitemap_urls(SITEMAP_LOCAL_PATH, SITEMAP_URL, days=DAYS_TO_FILTER)

    if not urls:
        urls = [
            'https://www.big-yellow-j.top/',
        ]
        print(f"Falling back to manual URL list: {urls}")

    max_workers = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_url, url) for url in urls]

    print("All submissions finished.")
