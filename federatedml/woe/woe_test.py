# coding=utf-8
'''
2021/10/12
author：ssw
unittest example
'''
import unittest
from federatedml.woe.woe_iv_ca_square_basic import *
from pip._vendor.distlib.compat import raw_input

#4定义测试类，父类为unittest.TestCase
class TestCaSquare(unittest.TestCase):
    # 所有类中方法的入参为self，定义方法的变量也要“self.变量”
    def setUp(self):
        self.former=10*np.random.random()
        self.latter=10*np.random.random()

    # 此处仅使用get_small展示了多个测试示例；需测试的方法可自行添加修改
    def test_case1(self):
        print('---------------get_small testing-------------')
        res=get_small('a',1)
        print(res)
        print('----------------case1 finished---------------')
    def test_case2(self):
        print('---------------get_small testing2-------------')
        print(self.former, self.latter)
        res2=get_small(self.former,self.latter)
        print(res2)
        print('----------------case2 finished---------------')
    def test_case3(self):
        print('---------------get_small testing3-------------')
        res3 = get_small([1,2,3], 1)
        print(res3)
        print('----------------case3 finished---------------')
    def test_case4(self):
        print('---------------get_small testing4-------------')
        self.number1=raw_input('enter number1')
        self.number1=int(self.number1)
        self.number2=raw_input('enter number2')
        self.number2=int(self.number2)
        res4= get_small(self.number1, self.number2)
        print(res4)
        print('----------------case4 finished---------------')
    #装饰器
    @unittest.skip('暂时跳过用例5的测试')
    def test_case5(self):
        print('---------------------装饰器-----------------')
        print(self.former)
        # assertEqual(a, b，[msg = '测试失败时打印的信息']):断言a和b是否相等，相等则测试用例通过。
        self.assertEqual(self.former,1005,msg='generated number is not 1005')
        print('----------------case5 finished---------------')
    # tearDown()方法用于测试用例执行之后的善后工作。
    def tearDown(self):
        print('test over')

if __name__ == '__main__':
    # 实例化测试套件
    suite=unittest.TestSuite()
    suite.addTest(TestCaSquare('test_case1'))
    suite.addTest(TestCaSquare('test_case2'))
    suite.addTest(TestCaSquare('test_case3'))
    suite.addTest(TestCaSquare('test_case4'))
    suite.addTest(TestCaSquare('test_case5'))
    runner=unittest.TextTestRunner()
    # 使用run()方法运行测试套件
    runner.run(suite)
