import inspect
import asyncio

class Test:
    async def foo(self):
        pass

print(inspect.iscoroutinefunction(Test.foo))
