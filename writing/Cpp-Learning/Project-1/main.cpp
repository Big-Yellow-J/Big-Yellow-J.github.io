#include "contact_action.h"
#include <iostream>
#include <string>
using namespace std;

#define MAX_PERSON 1000      // 最大人数

// g++ main.cpp code/contact_action.cpp code/util.cpp -I include -o contact_book && ./contact_book
int main() {
	int select = 0;
    ContactStrcut abs;
	abs.P_size = 0;

	// 随机用测试数据填充到 结构体里面
	abs.P_array[abs.P_size++] = PersonStruct{"张三", "138", "aabb", 1, 30};
	// abs.P_array[abs.P_size++] = PersonStruct{"1", "1", "1", 1, 1};
    abs.P_array[abs.P_size++] = PersonStruct{"李四", "139", "aabb", 0, 31};
    abs.P_array[abs.P_size++] = PersonStruct{"张三", "137", "aabb", 1, 29};

	SortContact(&abs);
	while (true)
	{
		ShowMenu();
		cin >> select;
		switch (select)
		{
		case 1:  //添加联系人
			AddPerson(&abs);
            break;
		case 2:  //显示联系人
            ShowPerson(&abs);
			break;
		case 3:  //删除联系人
            DeletePerson(&abs);
			break;
		case 4:  //查找联系人
			FindPerson(&abs);
            break;
		case 5:  //修改联系人
			ModifyPerson(&abs);
			break;
		case 6:  //清空联系人
			CleanPerson(&abs);
			break;
		case 0:  //退出通讯录
			cout << "欢迎下次使用" << endl;
			return 0;
			break;
		default:
			break;
		}
	}
	return 0;
}