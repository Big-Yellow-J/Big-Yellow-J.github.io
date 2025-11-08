#ifndef WORKER_MANGER_H
#define WORKER_MANGER_H

#include <iostream>
#include <vector>
#include <fstream>   // 写入文件
#include "employee.h"

#define FILENAME "empFILE.txt"

using namespace std;


class WorkerManager
{   
    private:
        vector<Worker*> workers; // 因为内部大部分都是通过多态处理的，对于多态需要使用指针
        bool m_FileisEmpty;      // 判断文件是不是空
    public:
        WorkerManager();
        ~WorkerManager();
        void show_menu();
        void exit_manager(); // 退出系统
        void add_people();   // 添加人员
        void save();         // 存储到txt中
        int get_people_num(); // 获取人数
        void show_people();  // 显示职工
        void del_people();   // 删除人员
        int is_exist(int id);// 检查是否存在
        void mod_people();   // 修改员工
};

#endif
