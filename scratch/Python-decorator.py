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

print('闭包的概念：https://www.cnblogs.com/s-1314-521/p/9763376.html')
print('总结：内存空间指针的应用')