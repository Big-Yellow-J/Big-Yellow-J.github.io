// contact_action.h
#ifndef CONTACT_ACTION_H
#define CONTACT_ACTION_H

#include <string>
#include <sstream>
#include <algorithm>   // std::stable_sort
using namespace std;

#define MAX_PERSON 1000

// 定义 人物信息 结构体 存储：1（string）、姓名、电话、地址；2（int）、性别、年龄
struct PersonStruct
{
    string P_name, P_phone, P_address;
    int P_sex, P_age;
};

// 定义 通讯录 结构体
struct ContactStrcut
{
    struct PersonStruct P_array[MAX_PERSON]; // 通讯录中保存的联系人数组
    int P_size;                               // 通讯录中人员个数
};

// 基础函数声明
int isExist(ContactStrcut* abs, string name);
void SortContact(ContactStrcut* abs);
void AddPersonInfo(PersonStruct* person);
bool ParseLineToPerson(string& line, PersonStruct& person);
// void MergeStruct_Pointer(ContactStrcut* tmp_struct, ContactStrcut* writed_struct);
// void MergeStruct_Cite(ContactStrcut& tmp_struct, ContactStrcut& writed_struct);
void MergeStruct(ContactStrcut* tmp_struct, ContactStrcut* writed_struct);
void MergeStruct(ContactStrcut& tmp_struct, ContactStrcut& writed_struct);

// 操作函数声明
void ShowMenu();
void AddPerson(ContactStrcut* abs);
void ShowPerson(ContactStrcut* abs);
void DeletePerson(ContactStrcut* abs);
void FindPerson(ContactStrcut* abs);
void ModifyPerson(ContactStrcut *abs);
void CleanPerson(ContactStrcut *abs);

#endif // CONCAT_ACTION_H