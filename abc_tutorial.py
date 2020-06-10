'''
ABC 클래스는 ABC 클래스를 상속받은 클래스(A)를 추후 다른 클래스가(B)가 상속받을 때,
A 클래스의 메서드를 반드시 구현하도록 강제하는 추상화 클래스다.

Q1 : __metaclass__란?
'''

import abc

class base:

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def func1(self):
        pass

    @abc.abstractmethod
    def func2(self):
        pass

class a(base):
    def func1(self):
        print("func1 a")
    def func2(self):
        print("func2 a")

class b(base):
    def func1(self):
        print("func1 b")
    # def func2(self):
    #     print("func2 b")

ma = a()
mb = b()

ma.func1()
ma.func2()
mb.func1()
mb.func2()