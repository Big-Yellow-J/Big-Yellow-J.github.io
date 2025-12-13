import time

def com_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        print(f"Used Time: {time.time() - start_time:.4f} 秒")  # 计算并输出执行时间
        return result
    return wrapper

@com_time
def test():
    time.sleep(0.5)
    print("Hello!")

def main():
    test()

# def main():
#     start_time = time.time()
#     test()
#     print(f"Used Time: {time.time()- start_time}")

class Test():
    def __init__(self, age):
        self.age = age
    
    def add(self):
        '''加一'''
        return self.age+ 1

# test = Test(13)
# test.__dict__['name'] = 'hjie'
# print(test.name)
# print(test.add.__name__)
# print(test.add.__doc__)


class Person:
    place= 'bj' # 类变量（所有实例共享）

    def __init__(self, name):
        self.name = name
    
    @staticmethod
    def age1(age):
        print(f"{age}")
    
    @classmethod
    def new_place(cls, new):
        cls.place = new

    def age2(self, age):
        print(f"{self.name}:{age} from {self.place}")
    
    def age3(self, age):
        if age>= 20:
            Person.new_place('cs')
        print(f"{self.name}:{age} from {self.__class__.place}")

Person.age1(13)
Person("Tom").age2(13)
Person("Tom").age3(23)

import torch

# with torch.no_grad()