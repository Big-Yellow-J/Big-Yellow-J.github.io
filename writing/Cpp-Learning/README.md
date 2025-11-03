# C++学习
## 基础语法学习
[C++基础知识](https://github.com/Blitzer207/C-Resource/blob/master/%E7%AC%AC1%E9%98%B6%E6%AE%B5C%2B%2B%20%E5%8C%A0%E5%BF%83%E4%B9%8B%E4%BD%9C%20%E4%BB%8E0%E5%88%B01%E5%85%A5%E9%97%A8/C%2B%2B%E5%9F%BA%E7%A1%80%E5%85%A5%E9%97%A8%E8%AE%B2%E4%B9%89/C%2B%2B%E5%9F%BA%E7%A1%80%E5%85%A5%E9%97%A8.md)

## 标准库学习
### sstream
允许你将字符串当作输入/输出流来使用，这使得从字符串中读取数据或将数据写入字符串变得非常简单。

```c++
#include <sstream>
#include <string>

// 从字符串读取数据
std::string data = "10 20.5";
std::stringstream ss(data);

int i;
double d;
ss >> i >> d; // 得到就是 (int)10 (double)20.5

// 向字符串写入数据
std::ostringstream oss;
int i = 100;
double d = 200.5;
oss << i << " " << d;
std::string result = oss.str(); // 得到就是 "100 200.5"
```
那么上述过程中就会自动将字符串 "10 20.5" **按照空格截断**然后分别将内容写入到i和d中其中i和d的格式按照最初定义一样。