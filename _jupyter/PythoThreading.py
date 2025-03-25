import multiprocessing

class SquareCalculator:
    def __init__(self, factor=2):
        self.factor = factor  # 使实例方法非静态，防止被 pickle

    def square(self, number):
        return number ** self.factor  # 这里 self 使其不可被 pickle

if __name__ == "__main__":
    calculator = SquareCalculator()  # 创建类实例
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    with multiprocessing.Pool(processes=4) as pool:
        results = pool.map(calculator.square, numbers)  # 这里会报错，无法 pickle
    print("原始列表:", numbers)
    print("平方结果:", results)
