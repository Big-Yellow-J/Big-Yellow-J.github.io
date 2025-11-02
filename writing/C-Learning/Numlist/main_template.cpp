/*
补充一（函数模板使用）：比如我需要排序数组但是数组不知道 类型 数组那么最开始使用 vector<int>...(vector<int>& num_list)
那么就会出现问题，那么就需要使用到 模板 方法来处理 <Numlist-Template> 都是使用模板方式

template <typename type> 
ret-type func-name(parameter list)
{
   // 函数的主体
}

*/

#include <iostream>
#include <vector>
#include <span>
#include <optional>
using namespace std;

void printf(const vector<double>& num_list) {
    for (const auto& x : num_list) {
        cout << x << " ";
    }
    cout << "\n";
}

template <typename list_T, typename target_T>
int sequence_search_template(std::span<list_T> num_list, target_T target) {
    /*
    对于num_list 不知道具体类型，那么直接使用 typename list_T 去根据输入函数自动判断 num_list
    的 typename
    */
    for (size_t i = 0; i < num_list.size(); ++i) {
        if (num_list[i] == target) {
            cout<< "找到数字，索引为："<< i<< "\n";
            return 0;
        }
    }
    cout<< "没有找到数字"<< "\n";
    return 0;
}

int main() {
    // g++ -std=c++20 main_template.cpp -o test
    auto num_list1 = vector<double>{2, 3, 5, 1, 10, 0, 2.2, 2.1, 5.2, 1.3, 10.0, 0.1};
    auto num_list2 = vector<int>{2, 3, 5, 1, 10, 0};
    sequence_search_template(std::span(num_list1), 3);
    sequence_search_template(std::span(num_list2), 30.0);
    return 0;
}