from warmup import wordle_difference

def test_identical() :
    assert wordle_difference('pizza', 'pizza') == (5,0)

def test_no_common() :
    assert wordle_difference('acorn', 'beige') == (0,0)

def test_anagrams() :
    assert wordle_difference('cares', 'scare') == (0,5)

def test_x() :
    assert wordle_difference('route', 'rebus') == (1,2)

def test_y() :
    assert wordle_difference('cocoa', 'about') == (0,2)

def test_z() :
    assert wordle_difference('learn', 'links') == (1,1)
