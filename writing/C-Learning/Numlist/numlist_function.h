#ifndef NUMLIST_H
#define NUMLIST_H

#include <vector>
#include <string>
using namespace std;

// 排序函数声明
vector<int> quick_sort_function(vector<int> num_list);
vector<int> insert_sort_function(vector<int> num_list);
vector<int> bubble_sort_function(vector<int> num_list);
void vector_print(vector<int> num_list, auto way);

// 查找函数声明
int sequence_search(vector<int> num_list, int target);
int binary_search(vector<int> num_list, int target);
int hash_search(vector<int> num_list, int target);
void search_print(int value);

#endif 