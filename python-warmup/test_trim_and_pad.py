from warmup import trim_and_pad

def test_tap1() :
    xx = []
    trim_and_pad(xx, 100)
    assert xx == []

def test_tap2() :
    xx = ['dog', 'cat', 'fly']
    trim_and_pad(xx, 3)
    assert xx == ['dog', 'cat', 'fly']

def test_tap3() :
    xx = ['dog', 'cat', 'fly']
    trim_and_pad(xx, 2)
    assert xx == ['do', 'ca', 'fl']

def test_tap4() :
    xx = ['dog', 'cat', 'fly']
    trim_and_pad(xx, 5)
    assert xx == ['dog  ', 'cat  ', 'fly  ']

def test_tap5() :
    xx = ['a', 'big', 'tree', 'grows', 'upward', 'forever']
    trim_and_pad(xx, 5)
    assert xx == ['a    ', 'big  ', 'tree ', 'grows', 'upwar', 'forev']
