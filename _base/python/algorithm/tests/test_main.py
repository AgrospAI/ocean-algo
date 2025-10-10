from pytest import raises

from src.algorithm import algorithm


def test_main():
    with raises(NotImplementedError):
        algorithm()
