import functools

int2=functools.partial(int,base =2)
print(int2('100001'))

max2= functools.partial(max,10)
print(max2(5,6,7))
print('='*20)

def add(*args,**kwargs):
    print(*args)
    print('-'*20)
    for key,value in kwargs.items():
        print('{}:{}'.format(key,value))
    return
add_partial=functools.partial(add,10,k1=10,k2=20)
print(add_partial(1,2,3,k5=40,k3=30))
#*args：10，1，2，3 *kwargs:update