from src.algorithm import algorithm


def test_main():
    algorithm()
    
    assert algorithm.result == 11

