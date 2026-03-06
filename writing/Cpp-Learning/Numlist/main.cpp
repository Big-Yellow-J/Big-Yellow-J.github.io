/*
C++ 中循环、判断、函数、数组、容器等
一、数组使用
数组基本定义
type arrayName [ arraySize ];


------------------------------------------------------------------------------------------------------

二、函数使用
一般函数定义就是：

返回类型 函数名称(变量名称){
    函数操作
}


补充一：多文件之间cpp使用
如果定义函数需要进行跨文件进行处理，那么可以按照下面方式处理（假设两个c++代码）
1、code-3.cpp；2、find_numlist.cpp；3、sort_numlist.cpp。那么处理步骤就是（假设要在1中使用2,3）
两种方式实现目标：1、使用头文件；2、使用makelist文件
第一种方式：
首先 定义头文件 numlist_function.h。在头文件中 把我需要使用的函数都加到头文件里面即可
而后 在所有需要”协同“的文件里面都添加 我的头文件 #include "numlist_function.h"
最后 编译。
上面过程中格外需要注意的是：⭐如果变量用auto处理的话要么将其函数放到 头文件中，要么将auto换成具体的类型

第二种方式：
直接写 CMakeLists.txt即可，然后不需要在每个文件里面定义头文件 #include "numlist_function.h"，不过
需要在最后的main.cpp 里面去使用 vector<int> insert_sort_function(vector<int> num_list);
编译方式的话：
cmake .
cmake --build .
然后运行 ./cmake_run

补充二：比如我需要排序数组但是数组不一定是一个 int 类型数组那么最开始使用 vector<int>...(vector<int>& num_list)
那么就会出现问题，那么就需要使用到 模板 方法来处理 <main_template.cpp> 都是使用模板方式
参考：1、https://juejin.cn/post/7078530622527897631
如果要跨文件使用（将<find_numlist.cpp>中的 sequence_search 换成模板写法），那么就需要将模板丢到头文件里面
⭐：需要格外注意如果使用模板，模板里面 const

补充二：在定义函数过程中有些会使用 &
// 1. 小类型：值传递
void print_int(int x) { ... }

// 2. 大对象 + 不修改：const 引用
void print_vector(const std::vector<int>& v) { ... }

// 3. 需要修改：非 const 引用
void increment(int& x) { x++; }

// 4. 通用容器求和（支持数组、vector 等）
template <typename Container>
int sum(const Container& data) { ... }  // 推荐！
------------------------------------------------------------------------------------------------------


TODO: 去学习如何使用 cmake规则语法
TODO: 去学习如何使用template模板定义函数 || 可以将 find_numlist 中所有内容都换成使用模板函数来定义
TODO: 去处理更加复杂情况
*/

#include <iostream>
#include <span>
#include <vector>            // 数组
#include <initializer_list>
#include <chrono>
#include <iomanip>

#include "algo/search.h"   // 使用 模板 函数要丢到头文件 h 中

using namespace std;
using namespace std::chrono;

// 排序函数声明
vector<int> quick_sort_function(vector<int>& num_list);
vector<int> insert_sort_function(vector<int>& num_list);
vector<int> bubble_sort_function(vector<int>& num_list);
void vector_print(vector<int>& num_list, auto way);

// 查找函数声明
int sequence_search(vector<int>& num_list, int target);
int binary_search(vector<int>& num_list, int target);
int hash_search(vector<int>& num_list, int target);
void search_print(int value);

// 使用 auto 参数编译过程中需要加参数 -std=c++20 
// g++ -std=c++20 code-3.cpp -o test
auto max_function(auto a, auto b) {
    if (a > b) return a;
    else return b;
}

template <typename T>
int sum(T& num_list){
    int tmp_value= 0;
    for (const auto& x : num_list) tmp_value += x;
    return tmp_value;
}

int main() {
    // 数组
    int num_list_tmp[] = {2, 3, 5, 1, 10, 0};
    cout <<"数组长度为："<< sizeof(num_list_tmp)/ sizeof(num_list_tmp[0])<< "\n";
    auto sum_value = sum(num_list_tmp);
    cout <<"数组之和为："<< sum_value<< "\n";

    // 定义容器
    auto num_list = vector<int>{2, 3, 5, 1, 10, 0};
    auto sum_value_vector = sum(num_list)                                                                                                                                                                                                                                         ;
    cout <<"输出 max_function(10, 20):\t" << max_function(10, 20) << "\n";
    cout <<"容器长度为："<< num_list.size()<< "\n";
    cout <<"容器之和为："<< sum_value<< "\n";

    // 插入排序处理
    insert_sort_function(num_list);
    // 冒泡排序处理
    bubble_sort_function(num_list);
    // 快速排序
    auto tmp_num_list = quick_sort_function(num_list);
    vector_print(tmp_num_list, "快速排序");

    // 暴力查找
    int index= sequence_search(tmp_num_list, 10);
    search_print(index);
    // 使用模板
    index = sequence_search_template(std::span(tmp_num_list), 10);
    search_print(index);
    // 二分查找
    // 第一次使用 int 已经定义了后续只需要 使用即可
    index= binary_search(tmp_num_list, 10);
    search_print(index);
    // 哈希查找
    index= hash_search(tmp_num_list, 10);
    search_print(index);

    return 0;
}