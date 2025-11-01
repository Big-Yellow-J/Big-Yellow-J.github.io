// #include "numlist_function.h"
#include <iostream>
#include <vector>     // 数组
using namespace std;

void vector_print(vector<int> num_list, auto way){
    cout<< "通过方式 "<< way<< " 得到结果为：\t";
    for (auto x : num_list) cout << x << " ";
    cout<< "\n";
}

vector<int> insert_sort_function(vector<int> num_list) {
    for (size_t i = 1; i < num_list.size(); i++) {
        auto index = num_list[i];
        size_t j = i;
        while (j > 0 && num_list[j-1] > index) {
            num_list[j] = num_list[j-1];
            --j;
        }
        num_list[j] = index;
    }
    vector_print(num_list, "插入排序");
    return num_list;
}

vector<int> bubble_sort_function(vector<int> num_list) {
    auto n = num_list.size();
    for (size_t i = 0; i < n; ++i) {
        bool swapped = false;
        for (size_t j = 0; j < n - i - 1; ++j) {
            if (num_list[j] > num_list[j + 1]) {
                swap(num_list[j], num_list[j + 1]);
                // 或：auto temp = num_list[j]; num_list[j] = num_list[j+1]; num_list[j+1] = temp;
                swapped = true;
            }
        }
        if (!swapped) break;
    }
    vector_print(num_list, "冒泡排序");
    return num_list;
}

vector<int> quick_sort_function(vector<int> num_list){
    if (num_list.size() <= 1) return num_list;

    auto pivot = num_list[0];
    vector<decltype(pivot)> left, right;

    for (size_t i = 1; i < num_list.size(); ++i) {
        if (num_list[i] <= pivot)
            left.push_back(num_list[i]);
        else
            right.push_back(num_list[i]);
    }

    // 递归排序左右 + 拼接
    left = quick_sort_function(left);
    right = quick_sort_function(right);

    // 合并：左边 + 基准 + 右边
    left.push_back(pivot);
    left.insert(left.end(), right.begin(), right.end());

    return left;
}
