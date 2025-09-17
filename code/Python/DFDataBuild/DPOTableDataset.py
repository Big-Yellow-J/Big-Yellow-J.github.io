import os
import re
import json
import html2text
from tqdm import tqdm
from bs4 import BeautifulSoup
from collections import defaultdict

def open_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def write_file(file_path, file_content):
    with open(file_path, 'w', encoding='utf-8') as f:
        if file_path.endswith('json'):
            pass
        else:
            f.write(file_content)

class DataBuildDPODataset():
    def __init__(self, json_path, image_path=None, random_ratio= [0.2, 0.4, 0.6, 0.8]):
        self.json_path = json_path
        self.image_path = image_path

        self.json_file = open_json(json_path)
    
    def html2md(self, html_file, md_path):
        '''将html转化为md'''
        md_file = html2text.html2text(html_file)
        write_file(md_path, md_file)
    
    def DPO_augment_table(self):
        '''
        对于html表格进行数据增强处理，主要用于构建DPO的数据集
        强化处理操作包括：1、随机行列交换；2、随机生成/取消部分线条；3、随机合并/拆分部分表格；4、随机给部分表格上“标记”
        处理流程：先去增强处理、最后格式化处理
        '''
        def format_html(html_file, type='basic'):
            '''格式化html
            1、去掉：<b> <i>标记，转换：<sub> <sub>标记为markdown数学公式符号 还需要去检查还有什么其他的数学公式表姐
            '''

            pass

        def tomarkdown(soup):
            '''将上下标、加粗、斜体转化为 Markdown/LaTeX 风格，并把数学公式用 $$ 包裹'''
            # 下标 <sub>
            for sub in soup.find_all("sub"):
                sub.string = f"_{{{sub.get_text()}}}"
                sub.unwrap()

            # 上标 <sub>
            for sup in soup.find_all("sup"):
                sup.string = f"^{{{sup.get_text()}}}"
                sup.unwrap()

            # 加粗 <b>
            # for b in soup.find_all("b"):
            #     b.string = f"****{b.get_text()}****"
            #     b.unwrap()

            # 斜体 <i>
            # for i in soup.find_all("i"):
            #     i.string = f"**{i.get_text()}**"
            #     i.unwrap()

            # 转成字符串
            html_text = str(soup)
            def wrap_math(match):
                return f"$${match.group(0)}$$"

            html_text = re.sub(r'\w+[_^]\{.*?\}', wrap_math, html_text)
            
            return html_text

        with tqdm(total=len(self.json_file)) as pbar:
            for i, (file_name, file_info) in enumerate(self.json_file.items()):
                html_file = file_info[0]['html']
                soup = BeautifulSoup(html_file, "html.parser")
                # self.html2md(html_content, f'./tmp/tmp-{i}.md')
                pbar.update(1)


if __name__ == '__main__':
    # data = open_json('./html_data/pubtabnet.json')
    # result = defaultdict(list)
    # for i, key in enumerate(data.keys()):
    #     if i<=10:
    #         result[key].append(data[key])
    # with open('./html_data/tmp_use.json', 'w', encoding= 'utf-8') as f:
    #     json.dump(result, f, ensure_ascii= True, indent=2)
    dpo_data = DataBuildDPODataset(json_path= './html_data/tmp_use.json')
    dpo_data.DPO_augment_table()