def decorator(func):
    def wrapper1(*args, ** kwargs):
        print('I am new added function')
        r=func(*args, ** kwargs)
        return r
    return wrapper1
@decorator
def deposit():
    print('depositing...')
def withdraw():
    print('withdrawing...')

button =1
if button ==1:
    deposit()
elif button ==2:
    withdraw()

print('-'*20,'\n')
print('带参数')
#TODO
#针对有些依赖函数签名的代码
import functools

def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper2(*args, ** kwargs):
                print('%s %s():'%(text,func.__name__))
                return func(*args, ** kwargs)
        return wrapper2
    return decorator

#TODO
#双层执行顺序
@decorator
@log('openwith')
def now():
    print('what time is it?')
now()
print(now.__name__)
@decorator
def yes(b):
    return b
#TODO
#一个小应用，打印分隔线
#Rough
def get_chars(func):
    def wrapper3():
        print('='*20)
        func()
    return wrapper3

def get_charss(func):
    def wrapper3():
        print('*'*20)
        func()
    return wrapper3

@get_charss
@get_chars
def now():
    print('what time is it?')
now()

#refined
def get_Char(a):
    def decorator(func):
        def wrapper():
            print(a*20)
            func()
        return wrapper
    return decorator

@get_Char('*')
@get_Char('#')
def now():
    print('what time is it?')
now()
#TODO
#闭包（closure)
print('闭包的概念：https://www.cnblogs.com/s-1314-521/p/9763376.html')
print('总结：内存空间指针的应用')
def lazy_sum(*args):
    def sum():
        s=0
        for x in args:
            s=s+x
        return s
    return sum

def createCounter():
    li = [0]
    def counter():
        li[0] += 1
        return li[0]
    return counter


counterA = createCounter()
print(counterA(), counterA(), counterA(), counterA(), counterA()) # 1 2 3 4 5
counterB = createCounter()
if [counterB(), counterB(), counterB(), counterB()] == [1, 2, 3, 4]:
    print('测试通过!')
else:
    print('测试失败!')

#利用闭包返回一个计数器函数，每次调用它返回递增整数
#函数的调用和执行，f,f()