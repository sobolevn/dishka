from dishka import Provider, Scope, make_container, provide
from dishka.provider import alias, decorate


class A:
    pass


class A1(A):
    pass


class A2(A1):
    pass


class ADecorator:
    def __init__(self, a: A):
        self.a = a


def test_simple():
    class MyProvider(Provider):
        a = provide(A, scope=Scope.APP)
        ad = decorate(ADecorator, provides=A)

    with make_container(MyProvider()) as container:
        a = container.get(A)
        assert isinstance(a, ADecorator)
        assert isinstance(a.a, A)


def test_alias():
    class MyProvider(Provider):
        a2 = provide(A2, scope=Scope.APP)
        a1 = alias(source=A2, provides=A1)
        a = alias(source=A1, provides=A)

        @decorate
        def decorated(self, a: A1) -> A1:
            return ADecorator(a)

    with make_container(MyProvider()) as container:
        a1 = container.get(A1)
        assert isinstance(a1, ADecorator)
        assert isinstance(a1.a, A2)

        a2 = container.get(A2)
        assert isinstance(a2, A2)
        assert a2 is a1.a

        a = container.get(A)
        assert a is a1
