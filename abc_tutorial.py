'''
ABC 클래스를 메타클래스로 지정해서 만든 클래스는 추상 클래스가 된다
    - 추상 클래스는 인스턴스로 만들 수 없다
    - @abstractmethod 데코레이터를 사용해서 반드시 구현해야 하는 메소드를 정의할 수 있다

Q1 : __metaclass__란?
metaclass를 나타낸다
metaclass란 클래스의 클래스다
python에서 클래스는 객체다
객체를 만드는 것이 클래스이기 때문에 클래스를 만들기 위한 클래스가 존재하고 이것이 메타클래스다
파이썬이 사용하는 내장된 메타클래스는 type이다
'''

from abc import *

class base(metaclass=ABCMeta):
    @abstractmethod
    def study(self):
        pass
    @abstractmethod
    def g2s(self):
        pass

class student(base):
    def study(self):
        print("study")

# 추상 클래스는 인스턴스로 만들 수 없다
mosi = sb()

# @abstractmethod 데코레이터를 사용해서 반드시 구현해야 하는 메소드를 정의할 수 있다
mosi = student()

'''
reference
1. https://dojang.io/mod/page/view.php?id=2389
2. https://tech.ssut.me/understanding-python-metaclasses/
'''

