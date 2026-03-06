#ifndef EMPLOYEE_H
#define EMPLOYEE_H

#include <iostream>
#include <string>

using namespace std;

// 职工抽象类
class Worker
{
    public:
        int m_id, m_deptid; // 职工的编号 部门名称
        string m_name;      // 职工的名称
        virtual void show_info()= 0;
        virtual string get_deptname()= 0;
        virtual ~Worker(){}
};

// 员工
class Employee: public Worker
{
    public:
        Employee(int id, int deptid, string name);
        virtual void show_info();
        virtual string get_deptname();
};

// 经理
class Manager: public Worker
{
    public:
        Manager(int id, int deptid, string name);
        virtual void show_info();
        virtual string get_deptname();
};

// 经理
class Boss: public Worker
{
    public:
        Boss(int id, int deptid, string name);
        virtual void show_info();
        virtual string get_deptname();
};
#endif
