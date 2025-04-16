---
layout: mypost
title: KnowledgeGraph基本原理以及数据库Neo4j使用
categories: 技术运用
extMath: true
images: true
address: changsha
show_footer_image: true
description: 主要介绍KnowledgeGraph基本原理及其应用
---


1、介绍KnowledgeGraph基本原理应用
2、实体抽取研究
3、**Neo4j**使用

## `Neo4j`使用

### Neo4j介绍


### Neo4j安装
**window**安装使用：https://blog.csdn.net/2301_77554343/article/details/135692019

> 1、如果不是安装桌面版本需要保证 **java**版本是支持 **Neo4j**的，[支持链接](https://neo4j.com/docs/upgrade-migration-guide/current/version-5/migration/breaking-changes/#:~:text=JDK%2017%20support%20and%20Scala,the%20Neo4j%20Database%205.14%20onwards.)
> 2、第一次使用需要更改初始密码：
> 2.1 启动neo4j（`neo4j console ` 或者 `neo4j start `）
> 2.2 输入初始密码（账号密码相同：neo4j）

### Neo4j使用
> 下面使用都是在Linux上进行操作

数据库一般就离不开：**增-删-差-改**这4个操作，在Neo4j中也同样如此。因此分别介绍在Neo4j中是如何进行这4类操作的（直接用默认的 电影数据库）
* 创建/删除用户

```bash
cypher-shell -u neo4j -p '当前密码'
:server user add
# 按提示输入：
用户名: your_new_user
密码: your_password
是否需要修改密码: false
```

浏览器端：

```
CREATE USER your_new_user SET PASSWORD 'your_password' CHANGE NOT REQUIRED;
GRANT ROLE admin TO your_new_user;  # 管理员权限
GRANT ROLE publisher TO your_new_user;  # 发布者权限
# 切换用户
:server switch-user
```

* 1、创建数据库

### Neo4j使用（主要结合Python进行使用）

## 参考
1、http://neo4j.com.cn/public/docs/chapter2/chapter2_1.html
