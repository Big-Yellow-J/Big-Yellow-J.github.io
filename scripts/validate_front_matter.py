"""校验 _posts/*.md 的 YAML front matter 完整性。

强制字段：title、categories、date（文件名/yaml 任一推出即可）。
缺失则 exit 1，让 GitHub Actions 早失败。
"""
import re
import sys
from pathlib import Path

import yaml

POSTS_DIR = Path('_posts')
REQUIRED = ('title', 'categories')
FILENAME_DATE = re.compile(r'^(\d{4}-\d{1,2}-\d{1,2})-')
FILENAME_DATE_STRICT = re.compile(r'^(\d{4}-\d{2}-\d{2})-')

errors: list[str] = []
warnings: list[str] = []

if not POSTS_DIR.is_dir():
    print(f'No {POSTS_DIR}/ directory, nothing to validate.')
    sys.exit(0)

for md in sorted(POSTS_DIR.glob('*.md')):
    text = md.read_text(encoding='utf-8')
    m = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if not m:
        errors.append(f'{md}: 缺少 YAML front matter')
        continue
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError as exc:
        errors.append(f'{md}: YAML 解析失败 - {exc}')
        continue

    for field in REQUIRED:
        if not meta.get(field):
            errors.append(f'{md}: 缺少必填字段 `{field}`')

    if not meta.get('date') and not FILENAME_DATE.match(md.name):
        errors.append(f'{md}: 既无 `date:` 字段也无 YYYY-MM-DD 文件名前缀')
    elif not meta.get('date') and not FILENAME_DATE_STRICT.match(md.name):
        warnings.append(f'{md}: 文件名日期非严格 YYYY-MM-DD 格式（建议补零，如 2025-03-04）')

    if not meta.get('description'):
        warnings.append(f'{md}: 无 description（optim_post.py 会自动补，但建议手填）')

for w in warnings:
    print(f'⚠️  {w}')

if errors:
    print('\n❌ Front matter 校验失败：')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)

print(f'\n✅ {len(list(POSTS_DIR.glob("*.md")))} 篇文章 front matter 全部合格。')
