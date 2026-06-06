from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import xml.etree.ElementTree as ET
import json
import os
import requests
from datetime import datetime, timedelta
import concurrent.futures

_proxy = os.getenv("BLOG_SUBMIT_PROXY")
if _proxy:
    os.environ["HTTP_PROXY"] = _proxy
    os.environ["HTTPS_PROXY"] = _proxy

SCOPES = ['https://www.googleapis.com/auth/indexing']
CREDENTIALS_FILE = './google_api.json'
SITE_URL = 'https://www.big-yellow-j.top/'
SITEMAP_LOCAL_PATH = '_site/static/xml/sitemap.xml'
SITEMAP_URL = 'https://www.big-yellow-j.top/static/xml/sitemap.xml'
DAYS_TO_FILTER = 50

BING_INDEXNOW_KEY = os.getenv('BING_INDEXNOW_KEY', 'd8eac6ca9eb040f28f6ed4ab5cd8d5ad')
BAIDU_PUSH_TOKEN = os.getenv('BAIDU_PUSH_TOKEN', '')


def _load_google_credentials():
    """优先用 GOOGLE_SA_JSON 环境变量（CI 用），fallback 到 google_api.json 文件（本地用）。"""
    sa_json = os.getenv('GOOGLE_SA_JSON')
    if sa_json:
        info = json.loads(sa_json)
        return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    if os.path.exists(CREDENTIALS_FILE):
        return service_account.Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPES)
    return None


credentials = _load_google_credentials()
service = build('indexing', 'v3', credentials=credentials) if credentials else None

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
    if service is None:
        return False
    body = {'url': url, 'type': 'URL_UPDATED'}
    try:
        service.urlNotifications().publish(body=body).execute()
        print(f"✅ Google submitted: {url}")
        return True
    except HttpError as error:
        print(f"❌ Google error {url}: {error.status_code} {error.reason}")
        return False
    except Exception as e:
        print(f"❌ Google other error {url}: {e}")
        return False

def bing_submit_url(url):
    endpoint = "https://api.indexnow.org/indexnow"
    domain = url.split('/')[2]
    payload = {
        "host": domain,
        "key": BING_INDEXNOW_KEY,
        "keyLocation": f'https://{domain}/Bing_indexnow.txt',
        "urlList": [url]
    }
    try:
        response = requests.post(endpoint, headers={"Content-Type": "application/json"}, json=payload, timeout=10)
        if response.status_code in (200, 202):
            print(f"✅ Bing submitted: {url}")
            return True
        print(f"❌ Bing failed {url}: {response.status_code} {response.text[:120]}")
        return False
    except Exception as e:
        print(f"❌ Bing exception {url}: {e}")
        return False

def baidu_submit_urls(urls):
    """百度普通收录 push（一次性批量），需 BAIDU_PUSH_TOKEN。"""
    if not BAIDU_PUSH_TOKEN or not urls:
        return False
    site = urls[0].split('/')[0:3]
    site_host = '/'.join(site).replace('https://', '').replace('http://', '')
    endpoint = f"http://data.zz.baidu.com/urls?site={site_host}&token={BAIDU_PUSH_TOKEN}"
    try:
        response = requests.post(endpoint, data='\n'.join(urls), headers={'Content-Type': 'text/plain'}, timeout=15)
        if response.status_code == 200:
            print(f"✅ Baidu submitted {len(urls)} urls: {response.text[:200]}")
            return True
        print(f"❌ Baidu failed: {response.status_code} {response.text[:200]}")
        return False
    except Exception as e:
        print(f"❌ Baidu exception: {e}")
        return False

def submit_url(url):
    """同时提交到 Google 和 Bing（百度走批量，在 main 里调用）"""
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

    max_workers = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(submit_url, url) for url in urls]
        concurrent.futures.wait(futures)

    baidu_submit_urls(urls)
    print("All submissions finished.")
