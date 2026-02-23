from pytest import raises

from debugger.algorithm.src.main import algorithm


def test_main():
    with raises(NotImplementedError):
        algorithm()
