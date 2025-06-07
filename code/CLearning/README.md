# C++ Learning
## 数组
> 所有代码：[⚙](./chapter04/)
**数组基本定义**： `typeNmae arrayName[arraySize]`。比如说下面几种定义方式都是合法的：

```c++
int tmp_array1[8]; // 直接定义 8个元素的数组
int tmp_array2[] = {0, 1, 2, 3}; // 不指定数组大小直接写入内容（内容数目就是数组大小）
int tmp_array2[10] = {0, 1, 2, 3};
```

**计算数组中元素个数**，可以直接通过：

```c++
sizeof tmp_array / sizeof tmp_array[0];
// 如果是字符数组
#include <cstring>
...
strlen(tmp_array);
```

`sizeof`:直接得到所占的字节大小；`strlen()`直接计算字符数组中的个数；


**向数组写入元素**，可以直接通过：

```c++
// 普通数组写入
int index = sizeof tmp_array / sizeof tmp_array[0]- 1; // 最后一个有效索引
cout << "Input new num: ";
cin >> tmp_array[index];
cout << "Updated num: " << tmp_array[index] <<endl;
// 面向行就行输入
cin.getline(tmp_array_3, sizeof tmp_array_3)
```

**值得注意的是**：使用 `cin`写入数组时候，在键盘输入遇到空格时候，输入内容会被“截断”只有空格前的内容会被写入到数组中，**空格后的内容就会停留在缓存中**。比如说下面代码中，理论上要两次用户输入，但是实际只进行了一次用户输入：

```c++
cout<< "Enter your name:n";
cin>> name;
Cout << "Enter your favorite dessert:\n";
cin>> dessert;
cout<< "I have some delicious"<< dessert;
cout<< "for you" << name;
```

那么直接使用 `cin` 写入代码的逻辑就是：
![](https://s2.loli.net/2025/06/07/qmWzuH3gveMG5LZ.png)

那么直接使用 `cin.getline` 写入代码逻辑就是：
![](https://s2.loli.net/2025/06/07/gn8uqGw7Ddj9Lam.png)

直接通过 cin 向数组写入字符串遇到 空格 就“停止”（只写入空格前的内容）因此可以通过
getline() 用通过回车键输入的换行符来确定输入结尾

值得注意的是，比如下面代码中使用 cin >> tmp_array[index]; 而后输入回车就会导致
在缓存区域里面存在 \n 因此通过 cin.ignore(); 进行去除。也就是说使用 cin 其实是在
缓存中不断写入，比如说通过cin： 99 99\n （\n代表输入回车）那么：\t（代表空格）99\n
还在缓存里面即使是 cin.ignore() 也就只能得到 99\n 导致这部分内容在继续写入接下来
的 cin操作（cin.getline(tmp_array_2, sizeof tmp_array_2);中）