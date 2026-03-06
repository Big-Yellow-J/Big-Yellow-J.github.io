#include "employee.h"

Employee::Employee(int id, int deptid, string name){
    m_id= id; m_deptid= deptid; m_name= name;
}
void Employee::show_info(){
    cout<< "职工编号："<< m_id
        << "\t 职工姓名："<< m_name
        << "\t 岗位："<< get_deptname()<<endl;
}
string Employee::get_deptname(){return string("员工");}

Manager::Manager(int id, int deptid, string name){
    m_id= id; m_deptid= deptid; m_name= name;
}
void Manager::show_info(){
    cout<< "职工编号："<< m_id
        << "\t 职工姓名："<< m_name
        << "\t 岗位："<< get_deptname()<<endl;
}
string Manager::get_deptname(){return string("经理");}

Boss::Boss(int id, int deptid, string name){
    m_id= id; m_deptid= deptid; m_name= name;
}
void Boss::show_info(){
    cout<< "职工编号："<< m_id
        << "\t 职工姓名："<< m_name
        << "\t 岗位："<< get_deptname()<<endl;
}
string Boss::get_deptname(){return string("老板");}