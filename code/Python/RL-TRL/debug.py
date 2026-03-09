def debug_function(a, b):
    tmp_list = []
    for i in range(1, b*a):
        tmp_list.append(i)
    print(sum(tmp_list))

if __name__=="__main__":
    debug_function(2, 3)