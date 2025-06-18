import requests
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

def parse_urls(input_string):
    # 使用正则表达式提取所有URL
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, input_string)
    return urls

def get_page_title(url):
    try:
        # 随机选择User-Agent以模拟不同浏览器
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59'
        ]
        headers = {
            'User-Agent': random.choice(user_agents)
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 尝试获取<title>标签内容
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.text.strip()
            
            # 清理标题中的多余部分
            if 'arxiv.org' in url:
                # 移除类似 [2311.05556] 的前缀（包括版本号如 [2406.14548v2]）
                title = re.sub(r'^\[\d{4}\.\d{5}(?:v\d+)?\]\s*', '', title)
                # 移除arXiv后缀
                title = re.sub(r'\s*\|\s*arXiv.*$', '', title)
            elif 'big-yellow-j.top' in url:
                # 移除Big Yellow J后缀
                title = re.sub(r'\s*-\s*Big Yellow J.*$', '', title)
            elif 'wrong.wang' in url:
                # 移除wrong.wang后缀
                title = re.sub(r'\s*-\s*Wrong Wang.*$', '', title)
            
            return (url, title)
        return (url, "No title found")
        
    except requests.RequestException as e:
        return (url, f"Error fetching {url}: {str(e)}")
    except Exception as e:
        return (url, f"Error processing {url}: {str(e)}")

def extract_titles(url_input):
    # 如果输入是字符串，解析出URL列表
    if isinstance(url_input, str):
        url_list = parse_urls(url_input)
        print(url_list)
    else:
        url_list = url_input
    
    results = []
    # 使用ThreadPoolExecutor进行并行处理
    with ThreadPoolExecutor(max_workers=2) as executor:
        # 提交所有URL任务
        future_to_url = {executor.submit(get_page_title, url): url for url in url_list}
        # 收集结果
        for future in as_completed(future_to_url):
            url, title = future.result()
            results.append(f"![{title}]({url})")
    return results

if __name__ == "__main__":
    urls = """
        [^1]:https://arxiv.org/abs/2303.01469
        [^2]:https://arxiv.org/abs/2310.04378
        [^3]:https://arxiv.org/abs/2402.19159
        [^4]:https://arxiv.org/pdf/2011.13456
        [^5]:https://arxiv.org/pdf/2406.14548v2
        [^6]:https://arxiv.org/abs/2311.05556
    """
    
    titles = extract_titles(urls)
    for info in titles:
        print(info)