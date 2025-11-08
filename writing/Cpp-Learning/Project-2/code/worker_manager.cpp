#include "worker_manager.h"
#include "employee.h"

WorkerManager::WorkerManager(){
	ifstream ifs;
	ifs.open(FILENAME, ios::in); // ios::in 打开文件读取
	if (!ifs.is_open())
	{
		cout<< "文件不存在"<< endl;
		m_FileisEmpty = true;
		ifs.close();
		return;
	}
	char ch;
	ifs>> ch;
	if (ifs.eof())
	{
		cout<< "文件为空！"<< endl;
		m_FileisEmpty = true;
		ifs.close();
		return;
	}

	ifs.clear();
    ifs.seekg(0, ios::beg);

    int id, deptid;
    string name;

    while (ifs >> id >> name >> deptid) {
        Worker* worker = nullptr;
        switch (deptid) {
            case 1: worker = new Employee(id, deptid, name); break;
            case 2: worker = new Manager(id, deptid, name); break;
            case 3: worker = new Boss(id, deptid, name); break;
        }
        if (worker) {
            workers.push_back(worker);
        }
    }

    ifs.close();
    m_FileisEmpty = workers.empty();
    cout << "成功加载 " << workers.size() << " 条员工记录。" << endl;
}
WorkerManager::~WorkerManager(){
	for (auto worker : workers) {
        delete worker;
    }
    workers.clear();
}
void WorkerManager::show_menu()
{
    cout << "********************************************" << endl;
	cout << "*********  欢迎使用职工管理系统！ **********" << endl;
	cout << "*************  0.退出管理程序  *************" << endl;
	cout << "*************  1.增加职工信息  *************" << endl;
	cout << "*************  2.显示职工信息  *************" << endl;
	cout << "*************  3.删除离职职工  *************" << endl;
	cout << "*************  4.修改职工信息  *************" << endl;
	cout << "*************  5.查找职工信息  *************" << endl;
	cout << "*************  6.按照编号排序  *************" << endl;
	cout << "*************  7.清空所有文档  *************" << endl;
	cout << "********************************************" << endl;
	cout << endl;
}
void WorkerManager::exit_manager(){cout<< "\n成功退出系统!\n"<<endl;}

void WorkerManager::add_people(){
    // 添加员工
	cout << "请选择要添加的职工类型：" << endl;
    cout << "1、普通员工" << endl;
    cout << "2、经理" << endl;
    cout << "3、老板" << endl;
    
    int choice;
    cin >> choice;
    
    if (choice < 1 || choice > 3) {
        cout << "输入错误！" << endl;
        return;
    }
    
    int id, deptId;
    string name;
    
    cout << "请输入职工编号：";
    cin >> id;
    cout << "请输入职工姓名：";
    cin >> name;
    
    Worker* worker = nullptr;
    
    switch (choice) {
        case 1:  // 普通员工
            worker = new Employee(id, 1, name);
            break;
        case 2:  // 经理
            worker = new Manager(id, 2, name);
            break;
        case 3:  // 老板
            worker = new Boss(id, 3, name);
            break;
    }
    
    if (worker) {
        workers.push_back(worker);
        cout << "添加成功！当前员工总数：" << workers.size() << endl;
    }
	save();
}

void WorkerManager::save(){
	ofstream ofs;
	ofs.open(FILENAME, ios::out); // ios::out 打开文件写入
	for (size_t i = 0; i < workers.size(); i++)
	{
		ofs<< workers[i]->m_id<< " " << workers[i]->m_name<< " "<< workers[i]->m_deptid<< endl;
	}
	ofs.close();
}

int WorkerManager::get_people_num(){
	ifstream ifs;
	ifs.open(FILENAME, ios::in);

	int id, dID;
	string name;
	int num = 0;
	while (ifs>> id && ifs>> name && ifs>> dID)
	{
		num ++;
	}
	ifs.close();
	return num;
}

void WorkerManager::show_people(){
	if (m_FileisEmpty){cout << "文件不存在或记录为空！" << endl;}
	else{
		for (size_t i = 0; i < workers.size(); i++)
		{
			workers[i]->show_info();
		}
	}
}

int WorkerManager::is_exist(int id){
	int index= -1;
	for (size_t i = 0; i < workers.size(); i++)
	{
		if (workers[i]->m_id== id)
		{
			return i;
		}		
	}
	return -1;
}

void WorkerManager::del_people(){
	if (m_FileisEmpty){cout << "文件不存在或记录为空！" << endl;}
	else{
		cout << "请输入想要删除的职工号：" << endl;
		int id = 0;
		cin >> id;

		int index= is_exist(id);
		if (index!= -1)
		{
			delete workers[index];
			workers[index] = nullptr;
			workers.erase(workers.begin() + index);
			cout<< "删除成功！！"<< endl;
		}
		save();
	}
}

void WorkerManager::mod_people(){
	if (m_FileisEmpty){cout << "文件不存在或记录为空！" << endl;}
	else{
		cout << "请输入修改职工的编号：" << endl;
		int id;
		cin >> id;

		int index = is_exist(id);
		if (index!= -1)
		{
			delete workers[index];
			workers[index] = nullptr;
			workers.erase(workers.begin() + index);
			add_people();
		}
		
	}
}