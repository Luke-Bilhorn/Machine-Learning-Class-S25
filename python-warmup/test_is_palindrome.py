from warmup import is_palindrome

def test_ip1() :
    assert is_palindrome([])

def test_ip2() :
    assert is_palindrome(['x'])

def test_ip3() :
    assert is_palindrome(['a','a'])

def test_ip4() :
    assert is_palindrome([x for x in 'racecar'])

def test_ip5() :
    assert is_palindrome([x for x in 'level'])

def test_ip6() :
    assert is_palindrome([x for x in 'abba'])

def test_ip7() :
    assert not (is_palindrome([x for x in 'boxcar']))

def test_ip8() :
    assert not (is_palindrome([x for x in 'ab']))
