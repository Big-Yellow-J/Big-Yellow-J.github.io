#include <iostream>
#include <vector>
#include <span>
#include <unordered_map>
using namespace std;

void search_print(int value){
    if (value !=-1)
    {
        cout<< "目标数据索引为：\t"<< value<< "\n";
    }else{
        cout<< "没有找到数据！";
    }
}

int sequence_search(vector<int>& num_list, int target){
    // 暴力查找
    for (size_t i = 0; i < num_list.size(); i++)
    {
        if (target== num_list[i])
        {
            return i;
        }
    }
    return -1;    
}

int binary_search(vector<int>& num_list, int target){
    // 二分查找算法
    int left= 0, right= num_list.size()-1;
    while (left<= right)
    {
        int mid= left+ (right- left)/2;
        if (num_list[mid]== target)
        {
            return mid;
        }else if (num_list[mid]< target)
        {
            left = mid+1;
        }else{
            right = mid- 1;
        }
    }
    return -1;
}

int hash_search(vector<int>& num_list, int target){
    unordered_map<int, int> dic;
    for (int i = 0; i < num_list.size(); i++) {
        int complement = target - num_list[i];
        if (dic.count(complement)) {
            return i;
        }
        dic[num_list[i]] = i;
    }
    return -1;
}

