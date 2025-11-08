#include "worker_manager.h"
#include "employee.h"
#include <iostream>
#include <sstream>

using namespace std;

// g++ main.cpp code/worker_manager.cpp -I include -o main && ./main
int main()
{
    WorkerManager wm;

	cout<< "人数为："<< wm.get_people_num()<< endl;
    int choice= 0;
    while (true)
    {
        wm.show_menu();
        cout << "请输入您的选择:" << endl;
        cin>> choice;
        switch (choice)
		{
		case 0: //退出系统
			wm.exit_manager();
            return 0;
		case 1: //添加职工
			wm.add_people();
			break;
		case 2: //显示职工
			wm.show_people();
			break;
		case 3: //删除职工
			wm.del_people();
			break;
		case 4: //修改职工
			wm.mod_people();
			break;
		default:
            cout<< "\n 输入错误!"<< endl;
			break;
		}
	}
    return 0;
}