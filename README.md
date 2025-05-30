主题来自：https://github.com/TMaize/tmaize-blog
# 本地化部署

## Window本地化部署
Wins上直接在wsl下进行使用：
第一步：
```bash
sudo apt install ruby-full build-essential ruby-bundler
```

第二步：

```bash
gem sources --add https://mirrors.tuna.tsinghua.edu.cn/rubygems/ --remove https://rubygems.org/
gem sources -l
gem sources --clear-all
gem sources --update
export GEM_HOME="~/.gems"
gem install bundler
bundle config mirror.https://rubygems.org https://mirrors.tuna.tsinghua.edu.cn/rubygems
bundle config list
bundle config set path "~/.gems"
```

通过下面命令启动/编译项目（进入到网页文件中）
```bash
cd Big-Yellow-J.github.io/
bundle install
bundle exec jekyll serve --watch --host=127.0.0.1 --port=8080
bundle exec jekyll build --destination=dist
```
