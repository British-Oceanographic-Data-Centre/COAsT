class Test:
    def test_method(self, *, arg1: str, arg2: str):
        print(arg1)
        print(arg2)


t1 = Test()
t1.test_method("test1", "test2") # This fails due to unexpected number of arguments.
t1.test_method(arg1 = "test1", arg2 = "test2") # It passes when using named arguments.
