#include "contact_action.h"
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

// 通讯录主界面
void ShowMenu(){
    cout << "***************************" << endl;
	cout << "*****  1、添加联系人  *****" << endl;
	cout << "*****  2、显示联系人  *****" << endl;
	cout << "*****  3、删除联系人  *****" << endl;
	cout << "*****  4、查找联系人  *****" << endl;
	cout << "*****  5、修改联系人  *****" << endl;
	cout << "*****  6、清空联系人  *****" << endl;
	cout << "*****  0、退出通讯录  *****" << endl;
	cout << "***************************" << endl;
}

// 添加联系人
void AddPerson(ContactStrcut *abs)
{
	if (abs->P_size== MAX_PERSON) return;
	
	ContactStrcut tmp_abs; // 建立 临时结构体 记录本次的输入
	tmp_abs.P_size = 0;

	int cin_way;
	cout<< "输入信息添加方式：1、直接复制粘贴整条；2、单条添加"<< endl;
	cin>> cin_way;
	cin.ignore(); // 清除换行符，防止 getline 读空

	if (cin_way== 1)
	{	
		cout << "请粘贴联系人信息（每行一人，格式：姓名 性别(1/2) 年龄 电话 地址），输入空行结束：" << endl;
        string line;
		while (getline(cin, line))
        {
            if (line.empty()) break;  // 空行结束输入
			
			PersonStruct person;
			if (ParseLineToPerson(line, person))
			{
				tmp_abs.P_array[tmp_abs.P_size++]= person;
			}
        }
	}
	else if (cin_way == 2)
	{
		AddPersonInfo(&tmp_abs.P_array[tmp_abs.P_size++]);
	}
	
	/*
	#mark: 对于两种合并方式：1、MergeStruct_Pointer通过 指针方式进行传递；2、MergeStruct_Cite通过通过 引用传递方式
	1、MergeStruct_Pointer 中两个参数本身都是 地址 而我的 abs 本身就是地址所以 tmp_abs 需要取地址
	2、MergeStruct_Cite    中两个参数本身都是 对象 而我的 abs 是地址所以 要通过 *abs 对指针abs解引用，得到它指向的对象

	指针函数：参数是*，调用时用&取地址
	引用函数：参数是&，调用时直接传对象

	C++支持函数重载，同样函数名称但是变量不同 编译器自动选择进行编译
	*/
	MergeStruct(tmp_abs, *abs); // MergeStruct_Cite
	MergeStruct(&tmp_abs, abs); // MergeStruct_Pointer
	// 最后排序
    SortContact(abs);
    cout << "本次操作完成！" << endl;
}

// 显示联系人
void ShowPerson(ContactStrcut *abs){
    if (abs-> P_size== 0)
    {
        cout << "当前记录为空" << endl;
    }
    else
    {
        for (size_t i = 0; i < abs->P_size; i++)
        {
            cout << "姓名：" << abs->P_array[i].P_name << "\t";
			cout << "性别：" << (abs->P_array[i].P_sex == 1 ? "男" : "女") << "\t";
			cout << "年龄：" << abs->P_array[i].P_age << "\t";
			cout << "电话：" << abs->P_array[i].P_phone << "\t";
			cout << "住址：" << abs->P_array[i].P_address << endl;
        }
    }
}

// 删除联系人
void DeletePerson(ContactStrcut *abs)
{
	string name;
	cout << "请输入您要删除的联系人" << endl;
	cin >> name;

	int ret = isExist(abs, name); // 注意 *abs 是形参 所以这里不要用 &abs
	if (ret != -1)
	{
		for (int i = ret; i < abs->P_size; i++)
		{
			abs->P_array[i] = abs->P_array[i + 1];
		}
         abs->P_size--;
		cout << "删除成功" << endl;
	}
	else
	{
		cout << "查无此人" << endl;
	}
}

// 查找联系人
void FindPerson(ContactStrcut *abs){
    string name;
    cout<< "需要查找的人的名称："<< endl;
    cin>> name;
    int ret = isExist(abs, name);
    if (ret== -1)
    {
        cout<< "查无此人"<< endl;
    }
    else
    {
        cout<< "姓名"<< abs->P_array[ret].P_name<< "\t";
        cout<< "性别"<< abs->P_array[ret].P_sex<< "\t";
        cout<< "年龄"<< abs->P_array[ret].P_age<< "\t";
        cout<< "电话"<< abs->P_array[ret].P_phone<< "\t";
        cout<< "住址"<< abs->P_array[ret].P_address<< endl;
    }
}

// 修改指定联系人信息
void ModifyPerson(ContactStrcut *abs){
	string name;
	cout<< "请输入您要修改的联系人" << endl;
	cin>> name;
	
	int ret= isExist(abs, name);
	if (ret== -1)
	{
		cout << "查无此人" << endl;
	}
	else
	{	
		int sex, age;
		string phone, address;
		cout<< "姓名："<< endl;
		cin>> name;
		abs-> P_array[ret].P_name= name;

		cout<< "输入性别:\n"<< "1 -- 男 \n 2 -- 女"<< endl;

		while (true)
		{
			cin >> sex;
			if (sex == 1 || sex == 2)
			{
				abs->P_array[ret].P_sex = sex;
				break;
			}
			else
			{
				cout << "输入有误，请重新输入"<< endl;
			}
		}

		cout << "请输入年龄：" << endl;
		cin >> age;
		abs->P_array[ret].P_age = age;

		cout << "请输入联系电话：" << endl;
		cin >> phone;
		abs->P_array[ret].P_phone = phone;

		//家庭住址
		cout << "请输入家庭住址：" << endl;
		cin >> address;
		abs->P_array[ret].P_address = address;

		cout << "修改成功" << endl;
	}
}

//清空所有联系人
void CleanPerson(ContactStrcut *abs){
	abs->P_size = 0;
	cout << "通讯录已清空" << endl;
}