#include "contact_action.h"
#include <iostream>
#include <string>

// 检查是否存在
int isExist(ContactStrcut* abs, string name)
{
    for (int i = 0; i < abs->P_size; i++) {
        if (abs->P_array[i].P_name == name) return i;
    }
    return -1;
}

// 自动排序
void SortContact(ContactStrcut* abs)
{
    if (!abs || abs->P_size <= 1) return;
    std::stable_sort(
        abs->P_array,                             // 开始迭代器
        abs->P_array + abs->P_size,               // 结束迭代器（开区间）
        [](const PersonStruct& a, const PersonStruct& b) -> bool {
            if (a.P_name != b.P_name)
                return a.P_name < b.P_name;
            if (a.P_age != b.P_age)
                return a.P_age < b.P_age;
            return a.P_phone < b.P_phone;
        }
    );
}

// 基础信息写入
void AddPersonInfo(PersonStruct* person)
{
    cout << "请输入姓名：" << endl;
    getline(cin, person->P_name);

    cout << "请输入性别：1--男 2--女" << endl;
    string sex_input;
    while (true)
    {
        getline(cin, sex_input);
        if (sex_input == "1") { person->P_sex = 1; break; }
        else if (sex_input == "2") { person->P_sex = 2; break; }
        else cout << "输入错误，请输入1或2：" << endl;
    }

    cout << "请输入年龄：" << endl;
    cin >> person->P_age;
    cin.ignore();  // 清除换行符

    cout << "请输入联系电话：" << endl;
    getline(cin, person->P_phone);

    cout << "请输入家庭住址：" << endl;
    getline(cin, person->P_address);
}

// 解析一整行的输入                            通过引用传递方式
bool ParseLineToPerson(string& line, PersonStruct& person)
{
    stringstream ss(line);
    string sex_str;

    // 顺序：姓名 性别(1/2) 年龄 电话 [地址...]
    if (!(ss >> person.P_name >> sex_str >> person.P_age >> person.P_phone))
        return false;

    if (sex_str == "1") person.P_sex = 1;
    else if (sex_str == "2") person.P_sex = 2;
    else return false;

    // 剩余部分作为地址（支持空格）
    string remaining;
    getline(ss, remaining);
    size_t start = remaining.find_first_not_of(" \t");
    person.P_address = (start == string::npos) ? "" : remaining.substr(start);

    return true;
}

// 将临时的 tmp_struct 写入到 writed_struct 中 通过指针传递方式
void MergeStruct(ContactStrcut* tmp_struct, ContactStrcut* writed_struct)
{
    for (size_t i = 0; i < tmp_struct->P_size; i++)
    {
        if (writed_struct->P_size >= MAX_PERSON) 
        {
            cout << "通讯录已满，停止添加" << endl;
            break;
        }

        const string name= tmp_struct->P_array[i].P_name;
        int pos= -1;
        for (int j = 0; j < writed_struct->P_size; ++j)
        {
            if (writed_struct->P_array[j].P_name == name)
            {
                pos = j;
                break;
            }
        }

        if (pos != -1)
        {
            cout << "姓名 '" << name << "' 已存在，是否覆盖？(y/n): ";
            char choice;
            cin >> choice;
            cin.ignore();
            if (choice == 'y' || choice == 'Y')
            {
                writed_struct->P_array[pos] = tmp_struct->P_array[i];
                cout << "已更新：" << name << endl;
            }
            else
            {
                cout << "跳过：" << name << endl;
            }
        }
        else
        {
            writed_struct->P_array[writed_struct->P_size++] = tmp_struct->P_array[i];
            cout << "添加成功：" << name << endl;
        }
    } 
}
// 将临时的 tmp_struct 写入到 writed_struct 中 通过引用传递方式
void MergeStruct(ContactStrcut& tmp_struct, ContactStrcut& writed_struct)
{
    for (size_t i = 0; i < tmp_struct.P_size; i++)
    {
        if (writed_struct.P_size>= MAX_PERSON)
        {
            cout << "通讯录已满，停止添加" << endl;
            break;
        }

        const string name= tmp_struct.P_array[i].P_name;
        int pos=1;
        for (size_t j = 0; j < writed_struct.P_size; j++)
        {
            if (writed_struct.P_array[j].P_name== name)
            {
                pos= j; break;
            }
            
        }
        if (pos != -1)
        {
            cout << "姓名 '" << name << "' 已存在，是否覆盖？(y/n): ";
            char choice;
            cin >> choice;
            cin.ignore();
            if (choice == 'y' || choice == 'Y')
            {
                writed_struct.P_array[pos] = tmp_struct.P_array[i];
                cout << "已更新：" << name << endl;
            }
            else
            {
                cout << "跳过：" << name << endl;
            }
        }
        else
        {
            writed_struct.P_array[writed_struct.P_size++] = tmp_struct.P_array[i];
            cout << "添加成功：" << name << endl;
        }
    }
}