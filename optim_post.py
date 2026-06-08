import os
import re
import time
import json
import yaml
import argparse
import requests
import threading
from PIL import Image
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv('./API_KEY.env')


class RateLimiter:
    """线程安全的最小间隔限速器，按 key 隔离（每个 token 一个独立时钟）。"""
    def __init__(self, qps: float):
        self._min_interval = 1.0 / qps if qps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = defaultdict(float)

    def wait(self, key: str):
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait_for = self._next_allowed[key] - now
            if wait_for > 0:
                time.sleep(wait_for)
                now = time.monotonic()
            self._next_allowed[key] = now + self._min_interval


_SMMS_LIMITER = RateLimiter(qps=1.5)
_SMMS_SEMAPHORE = threading.BoundedSemaphore(2)
_LLM_LIMITER = RateLimiter(qps=1.0)


def parse_smms_tokens(raw_tokens: str):
    """支持逗号/分号/换行分隔的多个 token。"""
    if not raw_tokens:
        return []
    items = re.split(r'[\n,;]+', raw_tokens)
    return [t.strip() for t in items if t.strip()]


def get_llm_config(provider: str):
    """按 provider 读取对应配置，支持 provider 专用键并回退到通用键。"""
    provider = provider.lower()
    if provider == 'deepseek':
        base_url = (
            os.getenv('DEEPSEEK_LLM_URL')
            or os.getenv('DEEPSEEK_BASE_URL')
            or 'https://api.deepseek.com'
        )
        api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('API_KEY')
        model_list = [
            'deepseek-v4-flash',
            'deepseek-v4-pro',
        ]
        return base_url, api_key, model_list

    base_url = os.getenv('DOUBAO_LLM_URL') or os.getenv('LLM_URL')
    api_key = os.getenv('DOUBAO_API_KEY') or os.getenv('API_KEY')
    model_list = [
        'doubao-seed-2-0-pro-260215',
        'doubao-seed-1-6-250615',
        'doubao-seed-1-6-flash-250615',
        'doubao-seed-1-8-251228',
        'doubao-seed-2-0-mini-260215',
        'doubao-1-5-pro-32k-250115',
        'doubao-1-5-lite-32k-250115'
    ]
    return base_url, api_key, model_list

def format_yaml_head(md_content):
    '''去除头部 YAML 标记'''
    yaml_pattern = r'^---\n.*?\n---\n'
    match = re.match(yaml_pattern, md_content, re.DOTALL)
    if match:
        yaml_str = md_content[match.start():match.end()].strip()
        yaml_dict = yaml.safe_load(yaml_str.replace('---', ''))
        md_content = md_content[match.end():].strip()
        return md_content, yaml_dict
    return md_content, None


def format_markdown(md_content,
                    format_way=['yaml_head', 'image', 'table', 'code', 'math', 'link']):
    '''格式化 Markdown 内容'''
    
    def format_image(md_content):
        '''替换图片标记为 <image>'''
        image_pattern = r'!\[[^\]]*\]\([^\)]+\)'
        return re.sub(image_pattern, '<图片>', md_content)
    
    def format_table(md_content):
        '''替换 Markdown 表格为 <table>（简单示例）'''
        table_pattern = r'\|.*?\|\n\|[-:\s]+\|\n(?:\|.*?\|\n)*'
        return re.sub(table_pattern, '<表格>', md_content, flags=re.MULTILINE)
    
    def format_code(md_content):
        '''替换代码块（行内和多行）为 <code>'''
        multiline_code_pattern = r'^```.*?\n.*?\n```'
        md_content = re.sub(multiline_code_pattern, '<代码>', md_content, flags=re.DOTALL | re.MULTILINE)
        # inline_code_pattern = r'`[^`]+`'
        # md_content = re.sub(inline_code_pattern, '<code>', md_content)
        return md_content
    
    def format_math(md_content):
        '''替换公式为 <math>'''
        block_math_pattern = r'\$\$.*?\$\$'
        md_content = re.sub(block_math_pattern, '<公式>', md_content, flags=re.DOTALL)
        inline_math_pattern = r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)'
        md_content = re.sub(inline_math_pattern, '<公式>', md_content)
        return md_content
    
    def format_link(md_content):
        '''替换链接标记为 <link>'''
        link_pattern = r'(?<!!)\[[^\]]*\]\([^\)]+\)'
        return re.sub(link_pattern, '<link>', md_content)
    
    yaml_dict = None
    for way in format_way:
        if way == 'yaml_head':
            md_content, yaml_dict = format_yaml_head(md_content)
        elif way == 'image':
            md_content = format_image(md_content)
        elif way == 'table':
            md_content = format_table(md_content)
        elif way == 'code':
            md_content = format_code(md_content)
        elif way == 'math':
            md_content = format_math(md_content)
        elif way == 'link':
            md_content = format_link(md_content)

    return md_content, yaml_dict


try:
    import pillow_avif  # noqa: F401
    _AVIF_AVAILABLE = True
except ImportError:
    _AVIF_AVAILABLE = False


def format_image(md_content, image_store_dir, max_threads, gen_avif=False):
    thread_local = threading.local()

    def get_session():
        session = getattr(thread_local, "session", None)
        if session is None:
            session = requests.Session()
            pool_size = max(8, max_threads * 2)
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=pool_size,
                pool_maxsize=pool_size
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            thread_local.session = session
        return session

    def upload_webp_file(webp_path: Path, image_bed='sm.ms'):
        """按 https://s.ee/docs/zh-CN/api/UploadFile/ 与 sm.ms v2 文档实现。

        - Authorization 头直接放 API key，无 Bearer / token 前缀
        - sm.ms 字段名为 smfile；s.ee 兼容端点字段名为 file（smfile 为别名）
        - 不传 domain（domain 是 s.ee 短链域名设置，错误的取值会被服务端拒绝）
        - 全局 Semaphore 限制同时上传 ≤ 2；同一 token 限速 ≤ 1.5 QPS
        - 429 / 5xx / 业务限流：指数退避重试最多 3 次
        - 401：跳过当前 token；其他业务错误：跳到下一个端点
        """
        if image_bed != 'sm.ms':
            return None
        smms_tokens = parse_smms_tokens(os.getenv("SMMS_API_LIST", ""))
        if not smms_tokens:
            print("❌ 上传失败：未设置 SMMS_API_LIST 环境变量，请检查 API_KEY.env")
            return None

        targets = [
            ('https://sm.ms/api/v2/upload', 'smfile'),
            ('https://s.ee/api/v1/file/upload', 'file'),
        ]
        session = get_session()

        def parse(resp):
            try:
                r = resp.json()
            except Exception:
                return None, f'HTTP {resp.status_code} 非 JSON'
            if r.get('success') or r.get('code') in (0, '0', 'success'):
                return (r.get('data') or {}).get('url'), None
            if r.get('code') == 'image_repeated':
                return r.get('images') or (r.get('data') or {}).get('url'), None
            return None, r.get('message') or f'code={r.get("code")}'

        for token in smms_tokens:
            mask = f"{token[:4]}***"
            for url, field in targets:
                for attempt in range(3):
                    _SMMS_LIMITER.wait(token)
                    with _SMMS_SEMAPHORE:
                        try:
                            with open(webp_path, 'rb') as f:
                                resp = session.post(
                                    url,
                                    files={field: f},
                                    headers={'Authorization': token},
                                    timeout=30,
                                )
                        except Exception as e:
                            print(f"⚠️  网络错误 [{mask} {url}] 第{attempt+1}次：{e}")
                            time.sleep(2 ** attempt)
                            continue
                    if resp.status_code == 401:
                        print(f"⚠️  token 鉴权失败（{mask}）：{url}")
                        break
                    if resp.status_code == 429 or resp.status_code >= 500:
                        print(f"⚠️  限流/5xx [{mask} {url}] HTTP{resp.status_code} 第{attempt+1}次")
                        time.sleep(2 ** attempt + 1)
                        continue
                    image_url, err = parse(resp)
                    if image_url:
                        return image_url
                    if err and re.search(r'flood|too many|rate|频繁|限制', err, re.I):
                        print(f"⚠️  业务限流（{mask}）：{err} 第{attempt+1}次")
                        time.sleep(2 ** attempt + 1)
                        continue
                    print(f"❌ 上传失败（{mask}）：{webp_path.name} - {err}")
                    break
        return None

    SKIP_HOSTS = ('s2.loli.net', 'i.loli.net', 'sm.ms', 's.ee')

    def download_convert(image_url, output_path, quality=85, image_bed='sm.ms'):
        """下载图片转 WebP（+ 可选 AVIF），返回 (源URL, webp_URL, avif_URL, 宽, 高)。"""
        host = urlparse(image_url).netloc
        if any(host.endswith(h) for h in SKIP_HOSTS):
            try:
                session = get_session()
                response = session.get(image_url, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                width, height = image.size
                print(f"⏭️  跳过转换（已是图床地址）：{image_url} ({width}x{height})")
                return image_url, image_url, None, width, height
            except Exception as e:
                print(f"⚠️  跳过但拉宽高失败：{image_url} - {e}")
                return image_url, image_url, None, 0, 0
        try:
            session = get_session()
            response = session.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            width, height = image.size

            output_path.parent.mkdir(parents=True, exist_ok=True)
            file_name_with_ext = Path(urlparse(image_url).path).name
            file_name = Path(file_name_with_ext).stem
            save_path = output_path / f"{file_name}.webp"
            image.save(save_path, format="webp", quality=quality)
            webp_url = upload_webp_file(save_path, image_bed=image_bed)

            avif_url = None
            if gen_avif and _AVIF_AVAILABLE:
                avif_path = output_path / f"{file_name}.avif"
                try:
                    image.save(avif_path, format="AVIF", quality=max(quality - 5, 50))
                    avif_url = upload_webp_file(avif_path, image_bed=image_bed)
                except Exception as e:
                    print(f"⚠️  AVIF 生成/上传失败（回退 WebP）：{e}")

            print(f"✅ 处理完成：{image_url} → {save_path.name} → {webp_url} ({width}x{height}){' +AVIF' if avif_url else ''}")
            return image_url, webp_url, avif_url, width, height
        except Exception as e:
            print(f"❌ 下载/转换失败：{image_url} - {e}")
            return image_url, None, None, None, None
    image_pattern = re.compile(
        r'!\[(?P<alt>[^\]]*)\]\((?P<url>https?://[^\s)]+\.(?:png|jpg|jpeg|webp|gif|bmp|svg))\)',
        re.IGNORECASE
    )
    matches = list(image_pattern.finditer(md_content))

    if not matches:
        return md_content

    urls = list({m.group('url') for m in matches})
    info_map = {}
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_url = {executor.submit(download_convert, u, image_store_dir): u for u in urls}
        for future in as_completed(future_to_url):
            image_url, webp_url, avif_url, w, h = future.result()
            if webp_url:
                info_map[image_url] = (webp_url, avif_url, w, h)

    if not info_map:
        return md_content

    def render(match):
        url = match.group('url')
        if url not in info_map:
            return match.group(0)
        webp_url, avif_url, w, h = info_map[url]
        alt = (match.group('alt') or 'image').replace('"', '&quot;')
        size_attr = f' width="{w}" height="{h}"' if w and h else ''
        if avif_url:
            return (
                f'<picture>'
                f'<source type="image/avif" srcset="{avif_url}">'
                f'<source type="image/webp" srcset="{webp_url}">'
                f'<img src="{webp_url}" alt="{alt}"{size_attr} loading="lazy" decoding="async" />'
                f'</picture>'
            )
        return f'<img src="{webp_url}" alt="{alt}"{size_attr} loading="lazy" decoding="async" />'

    md_content = image_pattern.sub(render, md_content)
    return md_content

def format_description(md_content, yaml_dict, description, md_path, provider='doubao'):
    '''生成摘要'''
    def llm_generate(md_content):
        '''通过llm生成描述'''
        base_url, api_key, model_list = get_llm_config(provider)
        if not base_url or not api_key:
            print(f"❌ LLM 配置缺失：provider={provider}，请检查 API_KEY.env")
            return None
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        messages = [{
            "role": "user",
            "content": f"""
# Role
你是一位资深技术编辑与 SEO 专家，擅长将复杂的 Markdown 技术文章重构为高信息密度、极具吸引力的中文摘要（Description）。

# Task
请基于我提供的 Markdown 内容，撰写一段用于全文预览的描述性摘要。

# Constraints
1. **零废话原则**：严禁使用“本文介绍了”、“作者认为”、“这篇文章探讨了”等引导词，直接输出文章核心逻辑与技术要点。
2. **纯净输出**：仅输出纯文本，禁止包含 Markdown 语法、引号、代码块、括号说明或任何解释性前缀/后缀。
3. **字数硬约束**：总字数必须严格控制在 150 到 250 个汉字之间，确保信息饱和度。
4. **技术精确性**：自然嵌入文章涉及的核心术语（如特定的算法名称、框架、参数等），保持专业语境。
5. **内容过滤**：自动忽略图片占位符、表格数据、代码片段及数学公式的原始表达，仅提取其背后的核心结论。

# Output Style
- **高信息量**：每一句话都必须承载实质性内容，避免空洞的修饰词（如“非常优秀”、“显著提升”等，改用具体的技术描述）。

# Input Data
请基于以下文章内容生成：{md_content}
        """
        }]

        re_connrct = 0
        while re_connrct< len(model_list):
            _LLM_LIMITER.wait(provider)
            try:
                completion = client.chat.completions.create(
                    model = model_list[re_connrct],
                    messages = messages
                )
                return completion.choices[0].message.content
            except Exception as e:
                msg = str(e)
                if re.search(r'429|rate|limit|quota|too many|频繁', msg, re.I):
                    time.sleep(2 ** min(re_connrct, 4) + 1)
                print(f"❌ LLM生成摘要出错（provider={provider}, model={model_list[re_connrct]}）：{e}")
                re_connrct+=1
        return None

    if yaml_dict.get('description', None) is None:
        old_md_content, yaml_dict = format_markdown(md_content)
        md_content, _ = format_yaml_head(md_content)
        llm_description = llm_generate(old_md_content)
        if llm_description:
            description[md_path] = [
                ('old_description', yaml_dict.get('description', None)),
                ('new_description', llm_description)]
            
            # 更新markdown的 yaml头标记
            yaml_dict['description'] = llm_description
            new_yaml_str = yaml.dump(yaml_dict, allow_unicode=True, sort_keys=False).strip()
            new_md_content = f"---\n{new_yaml_str}\n---\n\n{md_content.lstrip()}"
            return new_md_content, description
    return md_content, description

def process_file(file_path_list,
                 max_threads= None,
                 store_base= './images/post_image/',
                 description_path = './DEAL-MD.json',
                 provider='doubao',
                 file_threads=None,
                 image_threads=None,
                 desc_threads=None,
                 gen_avif=False):
    '''处理单个文件'''
    def mkdir_image_dir(md_path):
        md_path = Path(md_path)
        store_dir = Path(store_base) / md_path.stem.replace('-TODO', '')
        store_dir.mkdir(parents=True, exist_ok=True)
        return store_dir
    def open_file(path: str):
        if not os.path.exists(path):
            print(f"[WARN] 文件不存在: {path}")
            return None
        if os.path.getsize(path) == 0:
            print(f"[WARN] 文件为空: {path}")
            return "" if path.endswith(".md") else {}

        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith(".md"):
                return f.read()

            elif path.endswith(".json"):
                try:
                    return json.load(f)
                except json.JSONDecodeError as e:
                    print(f"[ERROR] JSON 解析失败: {path}")
                    print(e)
                    return {}
            
    cpu_count = os.cpu_count() or 1
    if max_threads is not None:
        file_threads = max_threads
        image_threads = max_threads
        desc_threads = max_threads
    else:
        file_threads = file_threads or min(max(2, cpu_count // 2), 8)
        image_threads = image_threads or 4
        desc_threads = desc_threads or 4

    file_threads = max(1, file_threads)
    image_threads = max(1, image_threads)
    desc_threads = max(1, desc_threads)

    print(
        f"⚙️ 并发配置: file_threads={file_threads}, "
        f"image_threads={image_threads}, desc_threads={desc_threads}"
    )
    file_description_dict = defaultdict(list)
    description = open_file(description_path) or {}

    md_paths = [p for p in file_path_list if p.endswith('.md')]

    def prepare_md(md_path):
        image_store_dir = mkdir_image_dir(md_path)
        md_content = open_file(md_path)
        if md_content is None:
            return None
        # md_content = format_image(md_content, image_store_dir, image_threads, gen_avif=gen_avif)
        _, yaml_dict = format_markdown(md_content)
        return md_path, md_content, yaml_dict

    with ThreadPoolExecutor(max_workers=file_threads) as executor:
        futures = {executor.submit(prepare_md, md_path): md_path for md_path in md_paths}
        for future in as_completed(futures):
            md_path = futures[future]
            try:
                result = future.result()
                if not result:
                    continue
                md_path, md_content, yaml_dict = result
                file_description_dict[md_path] = (md_content, yaml_dict)
            except Exception as e:
                print(f"❌ 预处理文件 {md_path} 出错: {e}")

    with ThreadPoolExecutor(max_workers=desc_threads) as executor:
        futures = {
            executor.submit(format_description, md_info[0], md_info[1], 
                            {}, md_path, provider): md_path
            for md_path, md_info in file_description_dict.items()
        }

        for future in as_completed(futures):
            md_path = futures[future]
            try:
                new_md_content, desc_update = future.result()
                description.update(desc_update)
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(new_md_content)
                print(f"✏️  Markdown 文件已更新：{md_path}")
            except Exception as e:
                print(f"❌ 处理文件 {md_path} 出错: {e}")

    with open(description_path, 'w', encoding= 'utf-8') as f:
        json.dump(description, f, indent= 2, ensure_ascii= False)
        
def main(post_dir_list = ['./_posts/', './writing/'],
         store_base= './images/post_image/',
         max_threads= None,
         provider='doubao',
         file_threads=None,
         image_threads=None,
         desc_threads=None,
         target_files=None,
         gen_avif=False):
    # 获取文件路径
    if target_files:
        md_path_list = target_files
    else:
        md_path_list = []
        for path in post_dir_list:
            for _ in os.listdir(path):
                md_path_list.append(os.path.join(path, _))
    process_file(md_path_list,
                 max_threads=max_threads,
                 store_base=store_base,
                 provider=provider,
                 file_threads=file_threads,
                 image_threads=image_threads,
                 desc_threads=desc_threads,
                 gen_avif=gen_avif)


def parse_args():
    parser = argparse.ArgumentParser(description='优化文章图片并生成摘要')
    parser.add_argument(
        '--provider',
        choices=['doubao', 'deepseek'],
        default='deepseek',
        help='摘要生成模型提供方，默认 deepseek'
    )
    parser.add_argument(
        '--max-threads',
        type=int,
        default=None,
        help='统一并发线程数（覆盖 file/image/desc 三类线程）'
    )
    parser.add_argument(
        '--file-threads',
        type=int,
        default=None,
        help='文件预处理并发线程数（默认自动）'
    )
    parser.add_argument(
        '--image-threads',
        type=int,
        default=None,
        help='单文件内图片下载/上传并发线程数（默认自动）'
    )
    parser.add_argument(
        '--desc-threads',
        type=int,
        default=None,
        help='摘要生成并发线程数（默认自动）'
    )
    parser.add_argument(
        '--store-base',
        default='./images/post_image/',
        help='图片本地缓存目录，默认 ./images/post_image/'
    )
    parser.add_argument(
        '--files',
        nargs='*',
        default=None,
        help='仅处理指定 Markdown 文件路径（可传多个）'
    )
    parser.add_argument(
        '--avif',
        action='store_true',
        default=False,
        help='额外生成 AVIF 格式并渲染为 <picture> 标签（需 pip install pillow-avif-plugin，会双倍占用图床配额）'
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(store_base=args.store_base,
         max_threads=args.max_threads,
         provider=args.provider,
         file_threads=args.file_threads,
         image_threads=args.image_threads,
         desc_threads=args.desc_threads,
         target_files=args.files,
         gen_avif=args.avif)
