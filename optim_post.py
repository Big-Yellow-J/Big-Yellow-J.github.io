import os
import re
import io
import json
import yaml
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from urllib.parse import urlparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv('./API_KEY.env')

def formad_markdown(md_content, 
                        format_way=['yaml_head', 'image', 'table', 'code', 'math', 'link']):
    '''格式化 Markdown 内容'''
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
def format_image(md_content, image_store_dir, max_threads):
    '''格式化图像'''
    def upload_webp_file(webp_path: Path, image_bed='sm.ms'):
        '''上传图片'''
        if image_bed == 'sm.ms':
            SMMS_API_LIST = os.getenv("SMMS_API_LIST", "").split(",")
            url = 'https://sm.ms/api/v2/upload'

            for token in SMMS_API_LIST:
                headers = {'Authorization': token}
                try:
                    with open(webp_path, 'rb') as f:
                        files = {
                            'smfile': (webp_path.name, f, 'image/webp')
                        }
                        response = requests.post(url, headers=headers, files=files, timeout=10)
                        response.raise_for_status()
                        result = response.json()

                        if result.get("success"):
                            return result["data"]["url"]
                        else:
                            print(f"❌ 上传失败（token: {token[:4]}***）：{webp_path.name} - {result.get('message')}")
                except Exception as e:
                    print(f"❌ 上传出错（token: {token[:4]}***）：{webp_path} - {e}")
            return None
        else:
            print("❌ 未支持的图床类型")
            return None
    def download_convert(image_url, output_path, quality=85, image_bed='sm.ms'):
        '''下载图片将其转化为webp'''
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")

            temp_buffer = io.BytesIO()
            image.save(temp_buffer, format='JPEG', quality=quality)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            file_name_with_ext = Path(urlparse(image_url).path).name
            file_name = Path(file_name_with_ext).stem
            save_path = output_path / f"{file_name}.webp"  
            image.save(save_path, format="webp", quality=quality)

            url_path = upload_webp_file(save_path, image_bed=image_bed)
            print(f"✅ 处理完成：{image_url} → {save_path.name} → {url_path}")
            return image_url, url_path
        except Exception as e:
            print(f"❌ 下载/转换失败：{image_url} - {e}")
            return image_url, None
    image_pattern = re.compile(
        r'!\[.*?\]\((https?://[^\s)]+\.(?:png|jpg|jpeg))\)', 
        re.IGNORECASE
    )
    matches = image_pattern.findall(md_content)
    # for image_url in matches:
    #     image_url, url_path = download_convert(image_url, image_store_dir)
    #     if url_path:
    #         md_content = md_content.replace(image_url, url_path)
    # return md_content
    future_to_url = {}
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for image_url in matches:
            future = executor.submit(download_convert, image_url, image_store_dir)
            future_to_url[future] = image_url

        for future in as_completed(future_to_url):
            image_url, url_path = future.result()
            if url_path:
                md_content = md_content.replace(image_url, url_path)

    return md_content

def format_description(md_content, yaml_dict, description, md_path):
    '''生成摘要'''
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

    def llm_generate(md_content):
        '''通过llm生成描述'''
        client = OpenAI(
            base_url= os.getenv("LLM_URL"),
            api_key= os.getenv("API_KEY")
        )
        messages = [{
            "role": "user",
            "content": f"""
                    你是一个专业的中文网站SEO描述生成助手，你的任务是根据我提供的Markdown文章内容，生成一段高质量的中文摘要，用于网页description。：

                    注意事项：
                    1. 输出内容必须是纯文本摘要，不需要任何开头说明，也不要包含引号、代码或其他解释性文字。
                    2. 字数控制在150到250个字符之间，简洁流畅，完整表达文章核心内容。
                    3. 摘要中自然融入文章的主题关键词和主要技术词汇，以便于搜索引擎优化（SEO）。
                    4. 忽略文章中的占位符（例如 <图片>、<表格>、<代码>、<数学公式> 等），无需描述这些内容。
                    5. 摘要必须紧凑有信息量，避免空洞、宽泛、重复或无意义的句子。
                    6. 你的摘要内容必须丰富！！

                    请基于以下文章内容生成高质量 description：
                    {md_content}
        """
        }]

        model_list = ['doubao-seed-1-6-250615', 'doubao-seed-1-6-flash-250615', 'doubao-seed-1-6-250615', 'doubao-1-5-lite-32k-250115']
        re_connrct = 0
        while re_connrct< len(model_list):
            try:
                completion = client.chat.completions.create(
                    model = model_list[re_connrct],
                    messages = messages
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"❌ LLM生成摘要上传出错：{e}")
                re_connrct+=1
        return None

    if yaml_dict.get('description', None) is None:
        old_md_content, yaml_dict = formad_markdown(md_content)
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
                 max_threads= 3,
                 store_base= './images/post_image/',
                 description_path = './DEAL-MD.json'):
    '''处理单个文件'''
    def mkdir_image_dir(md_path):
        md_path = Path(md_path)
        store_dir = Path(store_base) / md_path.stem.replace('-TODO', '')
        store_dir.mkdir(parents=True, exist_ok=True)
        return store_dir
    def open_file(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('md'):
                return f.read()
            elif path.endswith('json'):
                return json.load(f)
        
    max_threads = max(1, os.cpu_count() * 2) if os.cpu_count() is not None else max_threads
    file_description_dict = defaultdict(list)
    description = open_file(description_path)

    for md_path in file_path_list:
        if md_path.endswith('md'):
            image_store_dir = mkdir_image_dir(md_path)
            md_content = open_file(md_path)
            md_content = format_image(md_content, 
                                      image_store_dir,
                                      max_threads)
            _, yaml_dict = formad_markdown(md_content)
            file_description_dict[md_path] = (md_content, yaml_dict)

    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {
            executor.submit(format_description, md_info[0], md_info[1], 
                            {}, md_path): md_path
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
         max_threads= 3):
    # 获取文件路径
    md_path_list = []
    for path in post_dir_list:
        for _ in os.listdir(path):
            md_path_list.append(os.path.join(path, _))
    process_file(md_path_list, max_threads, 
                 store_base)

if __name__ == '__main__':
    main()