import functools


def call_log(m:str):
    def _w(func):
        @functools.wraps(func)
        def _inner_w(*args,**kwargs):
            print(f"[{m}]: {type(func)}: {func.__name__}: \n"
                  f"args={args}\n"
                  f"kwargs={kwargs}")
            ret = func(*args, **kwargs)
            print(f"[{m}] returns: {ret}\n"
                  f"{"~"*100}")
            return ret
        return _inner_w
    return _w

meta_call_log = call_log("meta")

class Meta(type):
    """Mimic the builtin `type`

    Notes:
        Instance creation flow:
        0. Meta creation: Meta.__new__ and Meta.__init__ at the class definition time (created with class MyClass:...)
        1. Class object creation: Meta.__call__ -> Meta.__new__ -> Meta.__init__ -> MyClass
        3. Intance creation: MyClass.__new__ -> MyClass.__init__

    References:
        https://jfreeman.dev/blog/2020/12/07/python-metaclasses/

    A class definition is a statement that constructs a `class object` and binds it to a variable in the local scope.

    A class object is a callable. Calling a class object typically returns an instance of that class.

    Every class object is an instance of its metaclass. The type of a class object is its metaclass.

    A class body is a block of statements, just like a function body. At the end of that block, the variables in the local scope become attributes of the class object.

    Metaclasses let us change the way a class definition constructs a class object. We can even make a class definition construct a value that is not a class object.

    Metaclasses let us change how instances of a class are constructed. We can even make a call of a class object return a value that is not an instance of that class.

    """

    @classmethod
    @meta_call_log
    def __prepare__(metacls, name, bases, **kwargs):
        """Call when class is define (class definition is a statement in python)"""
        assert issubclass(metacls, Meta)
        return (
            {}
        )  # can return a different type as long as it has __getitem__ and __setitem__.

    @meta_call_log
    def __new__(metacls, name, bases, namespace, **kwargs):
        """Construct a `class object` for a class whose metaclass is Meta.
        You can create class object directly.

        >>> MyClass = type('MyClass', (), {})
        >>> MyClass
        <class '__main__.MyClass'>

        """
        assert issubclass(metacls, Meta)
        cls = type.__new__(metacls, name, bases, namespace)
        return cls

    @meta_call_log
    def __init__(cls, name, bases, namespace, **kwargs):
        # Typically do nothing.
        assert isinstance(cls, Meta)

    @meta_call_log
    def __call__(cls, *args, **kwargs):
        """Construct an instance of a class whose metaclass is Meta.
        The cls is MyClass (The return of Meta.__new__)
        """
        assert isinstance(cls, Meta) and cls is not Meta # Myclass is the instance of Meta, but not Meta.
        
        # Note: new and init will be pass the same args and kwargs. No matter you use or not.
        obj = cls.__new__(cls, *args, **kwargs)
        if isinstance(obj, cls):
            cls.__init__(obj, *args, **kwargs) # Myclass.__init__
        return obj

class_call_log = call_log("class")

class MyClass(metaclass=Meta):
    
    @class_call_log
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)
    
    @class_call_log
    def __init__(self,*args,**kwargs):
        self.a = 5
    

if __name__ == "__main__":
    print("main start")
    myclass = MyClass("my class arguments")
    print(f"main end: {myclass.a}")
    
    """
Console:

/home/hieu/Workspace/projects/petorch/.venv/bin/python /home/hieu/Workspace/projects/petorch/draft/meta.py
[meta]: <class 'function'>: __prepare__:
args=(<class '__main__.Meta'>, 'MyClass', ())
kwargs={}
[meta] returns: {}
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[meta]: <class 'function'>: __new__:
args=(<class '__main__.Meta'>, 'MyClass', (), {'__module__': '__main__', '__qualname__': 'MyClass', '__new__': <function MyClass.__new__ at 0x766fd12a27a0>, '__init__': <function MyClass.__init__ at 0x766fd12a28e0>, '__classcell__': <cell at 0x766fd12acbb0: empty>})
kwargs={}
[meta] returns: <class '__main__.MyClass'>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[meta]: <class 'function'>: __init__:
args=(<class '__main__.MyClass'>, 'MyClass', (), {'__module__': '__main__', '__qualname__': 'MyClass', '__new__': <function MyClass.__new__ at 0x766fd12a27a0>, '__init__': <function MyClass.__init__ at 0x766fd12a28e0>, '__classcell__': <cell at 0x766fd12acbb0: Meta object at 0x246164f0>})
kwargs={}
[meta] returns: None
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main start
[meta]: <class 'function'>: __call__:
args=(<class '__main__.MyClass'>, 'my class arguments')
kwargs={}
[class]: <class 'function'>: __new__:
args=(<class '__main__.MyClass'>, 'my class arguments')
kwargs={}
[class] returns: <__main__.MyClass object at 0x766fd12accb0>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[class]: <class 'function'>: __init__:
args=(<__main__.MyClass object at 0x766fd12accb0>, 'my class arguments')
kwargs={}
[class] returns: None
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
[meta] returns: <__main__.MyClass object at 0x766fd12accb0>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main end: 5

Process finished with exit code 0

    """