#!/usr/bin/env python3
"""
将 .docx 文件转换为 Markdown 格式。
用法:
    python doc_to_md.py <input.docx> [output.md]

依赖:
    pip install python-docx
"""

import sys
import os
import re
from pathlib import Path

try:
    from docx import Document
    from docx.oxml.ns import qn
except ImportError:
    print("错误: 请先安装 python-docx: pip install python-docx")
    sys.exit(1)


def extract_paragraph_text(para) -> str:
    """提取段落文本，处理超链接等复杂元素。"""
    # 检查是否包含超链接
    hyperlinks = para._element.findall(
        './/' + qn('w:hyperlink')
    )
    if not hyperlinks:
        return para.text

    # 重构包含超链接的段落文本
    full_text = []
    for run_elem in para._element.iter():
        tag = run_elem.tag.split('}')[-1] if '}' in run_elem.tag else run_elem.tag
        if tag == 't':
            text = run_elem.text or ''
            full_text.append(text)
        elif tag == 'tab':
            full_text.append('\t')
        elif tag == 'br':
            full_text.append('\n')

    return ''.join(full_text)


def extract_images(doc: Document, output_dir: str) -> dict:
    """提取文档中的图片，保存到指定目录，返回 {rId: 文件名} 的映射。"""
    image_map = {}
    if not hasattr(doc, 'part'):
        return image_map

    os.makedirs(output_dir, exist_ok=True)
    image_count = 0

    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            image_count += 1
            image = rel.target_part
            ext = os.path.splitext(image.partname)[-1] or '.png'
            image_name = f"image_{image_count}{ext}"
            image_path = os.path.join(output_dir, image_name)
            with open(image_path, 'wb') as f:
                f.write(image.blob)
            image_map[rel.rId] = image_name

    return image_map


def detect_heading_level(para) -> int:
    """检测段落的标题级别（1-6），如果不是标题则返回 0。"""
    style_name = (para.style.name if para.style else '').lower()

    # 检查样式名称
    heading_patterns = [
        (r'heading\s*1|heading1|标题\s*1', 1),
        (r'heading\s*2|heading2|标题\s*2', 2),
        (r'heading\s*3|heading3|标题\s*3', 3),
        (r'heading\s*4|heading4|标题\s*4', 4),
        (r'heading\s*5|heading5|标题\s*5', 5),
        (r'heading\s*6|heading6|标题\s*6', 6),
    ]
    for pattern, level in heading_patterns:
        if re.search(pattern, style_name):
            return level

    # 通过大纲级别检测（直接从 XML 属性读取）
    pPr = para._element.find(qn('w:pPr'))
    if pPr is not None:
        outline_elem = pPr.find(qn('w:outlineLvl'))
        if outline_elem is not None:
            try:
                val = int(outline_elem.get(qn('w:val'), '0'))
                return min(val + 1, 6)
            except (ValueError, TypeError):
                pass

    return 0


def detect_list_info(para) -> tuple:
    """检测段落是否为列表项，返回 (是否为列表, 列表级别, 编号)。"""
    numPr = para._element.find('.//' + qn('w:numPr'))
    if numPr is None:
        return False, 0, ''

    # 获取列表级别
    ilvl_elem = numPr.find(qn('w:ilvl'))
    level = int(ilvl_elem.get(qn('w:val'), '0')) if ilvl_elem is not None else 0

    # 提取当前编号文本
    num_text = ''
    # 尝试从段落中提取自动编号
    num_id = numPr.find(qn('w:numId'))
    if num_id is not None:
        # 编号文本可能已嵌入段落头部，尝试获取
        pass

    return True, level, num_text


def detect_bold(para) -> bool:
    """检测段落是否主要是粗体。"""
    runs = para.runs
    if not runs:
        return False
    bold_count = sum(1 for r in runs if r.bold)
    return bold_count >= len(runs) * 0.5 and bold_count > 0


def detect_italic(para) -> bool:
    """检测段落是否主要是斜体。"""
    runs = para.runs
    if not runs:
        return False
    italic_count = sum(1 for r in runs if r.italic)
    return italic_count >= len(runs) * 0.5 and italic_count > 0


def get_run_formatting(run) -> tuple:
    """获取 run 的格式标记：返回 (前缀, 后缀)。"""
    prefix = ''
    suffix = ''
    if run.bold:
        prefix += '**'
        suffix += '**'
    if run.italic:
        prefix += '*'
        suffix += '*'
    if hasattr(run, 'underline') and run.underline:
        prefix += '<u>'
        suffix += '</u>'
    if hasattr(run, 'font') and run.font.strike:
        prefix += '~~'
        suffix += '~~'
    return prefix, suffix


def para_to_markdown(para, image_map: dict, image_dir: str) -> str:
    """将单个段落转换为 Markdown 文本。"""
    text = extract_paragraph_text(para).strip()

    # ---- 图片处理 ----
    drawings = para._element.findall('.//' + qn('w:drawing'))
    if drawings:
        md_lines = []
        for dw in drawings:
            blip = dw.find('.//' + qn('a:blip'))
            if blip is not None:
                embed = blip.get(qn('r:embed'))
                if embed and embed in image_map:
                    img_name = image_map[embed]
                    img_path = os.path.join(image_dir, img_name)
                    alt_text = text or '图片'
                    md_lines.append(f'![{alt_text}]({img_path})')
            else:
                md_lines.append(f'![{text or "图片"}](image_placeholder)')
        return '\n\n'.join(md_lines) + '\n'

    if not text and not drawings:
        return '\n'

    # ---- 标题检测 ----
    heading_level = detect_heading_level(para)

    # ---- 列表检测 ----
    is_list, list_level, _ = detect_list_info(para)
    indent = '  ' * list_level

    # ---- 表格检测（简化处理） ----
    # 表格在 docx 中需要特殊处理，此处仅支持段落

    # ---- 构建 Markdown ----

    # 标题
    if heading_level:
        return f"{'#' * heading_level} {text}\n\n"

    # 列表
    if is_list:
        prefix = '- ' if list_level == 0 else '  - '
        if text:
            return f"{indent}{prefix}{text}\n"
        return '\n'

    # 粗体检测：如果整段主要是粗体且较短，可能是不规范的标题
    if detect_bold(para) and len(text) < 80 and not text.endswith(('。', '；', '，')):
        return f"**{text}**\n\n"

    # 普通段落 — 处理内联格式
    runs = para.runs
    if runs and any(r.bold or r.italic for r in runs):
        md_parts = []
        for run in runs:
            t = run.text or ''
            if not t:
                continue
            prefix, suffix = get_run_formatting(run)
            md_parts.append(f'{prefix}{t}{suffix}')
        return ''.join(md_parts) + '\n\n'
    else:
        return text + '\n\n'


def extract_table_to_markdown(table) -> str:
    """将 docx 表格转换为 Markdown 表格。"""
    if not table.rows:
        return ''

    md_lines = []
    rows = list(table.rows)

    for i, row in enumerate(rows):
        cells = [
            cell.text.replace('\n', ' ').replace('|', '\\|').strip()
            for cell in row.cells
        ]
        md_lines.append('| ' + ' | '.join(cells) + ' |')

        # 表头后添加分隔行
        if i == 0:
            md_lines.append('| ' + ' | '.join(['---'] * len(cells)) + ' |')

    return '\n'.join(md_lines) + '\n\n'


def iter_block_items(parent):
    """
    生成器：依次产出 Document 的块级元素（段落和表格），保持文档顺序。
    参考 python-docx 官方文档示例。
    """
    from docx.document import Document as _Document
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.table import _Cell, Table
    from docx.text.paragraph import Paragraph

    if isinstance(parent, _Document):
        body = parent.element.body
    elif isinstance(parent, _Cell):
        body = parent._tc
    else:
        raise TypeError(f"不支持的父元素类型: {type(parent)}")

    for child in body:
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def convert_docx_to_markdown(input_path: str, output_path: str = None) -> str:
    """主转换函数：将 .docx 文件转换为 Markdown。"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    doc = Document(input_path)

    # 提取图片
    base_name = Path(input_path).stem
    image_dir = f"{base_name}_images"
    image_map = extract_images(doc, image_dir)

    md_content = []

    for item in iter_block_items(doc):
        if hasattr(item, 'text') and not hasattr(item, 'rows'):
            # 段落
            md_content.append(para_to_markdown(item, image_map, image_dir))
        elif hasattr(item, 'rows'):
            # 表格
            md_content.append(extract_table_to_markdown(item))

    result = ''.join(md_content)

    # 清理多余的空行
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = result.strip() + '\n'

    # 写入输出文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"✓ 转换完成: {input_path} -> {output_path}")
        if image_map:
            print(f"✓ 图片已提取到: {image_dir}/ ({len(image_map)} 张)")

    return result


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("用法: python doc_to_md.py <input.docx> [output.md]")
        sys.exit(1)

    input_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = Path(input_path).with_suffix('.md')

    try:
        convert_docx_to_markdown(input_path, output_path)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
