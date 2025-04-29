from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import xml.etree.ElementTree as ET
import os
import requests
from datetime import datetime, timedelta
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

SCOPES = ['https://www.googleapis.com/auth/indexing']
CREDENTIALS_FILE = './google_api.json'
SITE_URL = 'https://www.big-yellow-j.top/'  # 你的网站 URL
SITEMAP_LOCAL_PATH = '_site/static/xml/sitemap.xml'  # 本地 Jekyll 构建的 sitemap 路径
SITEMAP_URL = 'https://www.big-yellow-j.top/static/xml/sitemap.xml'  # 线上 sitemap URL
DAYS_TO_FILTER = 7  # 只提交最近 7 天更新的 URL（可调整）

credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES)
service = build('indexing', 'v3', credentials=credentials)

def get_sitemap_urls(local_path, online_url, days=None):
    """从本地文件或线上 URL 获取 sitemap 的 URL，可选根据 lastmod 过滤"""
    content = None
    source = None

    # 优先尝试本地文件
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

    # 如果本地文件不可用，尝试线上 URL
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

    try:
        tree = ET.fromstring(content)  # 解析 XML
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = []

        if days is not None:
            cutoff_date = datetime.now() - timedelta(days=days)

        for url_elem in tree.findall('.//ns:url', namespace):
            loc = url_elem.find('ns:loc', namespace)
            lastmod = url_elem.find('ns:lastmod', namespace)

            if loc is None:
                continue

            url = loc.text
            if days is None:
                urls.append(url)
            elif lastmod is not None:
                try:
                    lastmod_date = datetime.strptime(lastmod.text, '%Y-%m-%d')
                    if lastmod_date >= cutoff_date:
                        urls.append(url)
                except ValueError as e:
                    print(f"Invalid lastmod format for {url}: {e}")
                    urls.append(url)  # 如果日期格式错误，仍包含 URL

        return urls
    except ET.ParseError as e:
        print(f"Error parsing sitemap: {e}")
        print(f"Sitemap content preview:\n{content[:500]}")
        return []
    except Exception as e:
        print(f"Unexpected error while parsing sitemap: {e}")
        return []

def google_submit_url(url):
    """提交单个 URL 到 Google Indexing API"""
    body = {
        'url': url,
        'type': 'URL_UPDATED'  # 或 'URL_DELETED' 如果需要删除索引
    }
    try:
        response = service.urlNotifications().publish(body=body).execute()
        print(f"✅ Submitted {url}: {response}")
        return True
    except HttpError as error:
        print(f"❌ Error submitting {url}: {error.status_code} {error.reason}")
        return False
    except Exception as e:
        print(f"❌ Other error submitting {url}: {e}")
        return False

def bing_submit_url(url):
    endpoint = "https://api.indexnow.org/indexnow"

    domain = url.split('/')[2]  # 例如 www.big-yellow-j.top
    payload = {
        "host": domain,
        "key": 'd8eac6ca9eb040f28f6ed4ab5cd8d5ad',
        "keyLocation": 'https://www.big-yellow-j.top/Bing_indexnow.txt',  # 这里必须是完整URL
        "urlList": [url]  # 注意！是完整的url，不要截断
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(endpoint, headers=headers, json=payload)
        if response.status_code in [200, 202]:
            print(f"✅ Successfully submitted to IndexNow: {url}")
            return True
        else:
            print(f"❌ Failed to submit to IndexNow: {url}, Status Code: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Exception submitting to IndexNow: {e}")
        return False

if __name__ == '__main__':
    # 从 sitemap 获取 URL，过滤最近 DAYS_TO_FILTER 天的更新
    urls = get_sitemap_urls(SITEMAP_LOCAL_PATH, SITEMAP_URL, days=DAYS_TO_FILTER)

    if not urls:
        urls = [
            'https://www.big-yellow-j.top/',
        ]
        print(f"Falling back to manual URL list: {urls}")
    success_count = 0
    for url in urls:
        try:
            google_submit_url(url)
            bing_submit_url(url)
        except Exception:
            continue