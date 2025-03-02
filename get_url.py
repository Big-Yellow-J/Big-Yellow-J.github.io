import os

path_dir = './_posts/'
for name in os.listdir(path_dir):
    tmp = ''
    for _ in name.replace('.md', '').split('-')[:3]:
        tmp += f"/{_}"
    md_name = name[11:-3]
    tmp = f"{tmp}/{md_name}.html"

    url = f"https://www.big-yellow-j.top/posts{tmp}"
    print(url)
    # print(name)