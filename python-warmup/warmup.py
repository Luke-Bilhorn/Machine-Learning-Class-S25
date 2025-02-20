import copy

def selection_sort(sequence):
    for i in range(len(sequence)):
        least = sequence[i]
        leastIndex = i
        for j in range(i, len(sequence)):
            if sequence[j] < least:
                least = sequence[j]
                leastIndex = j
        temp = sequence[i]
        sequence[i] = least
        sequence[leastIndex]= temp

def count_common(xx, yy) :
    selection_sort(xx)
    xx = set(xx)
    selection_sort(yy)
    yy = set(yy)
    return len(xx.intersection(yy))

def reverse(xx): 
    for i in range(len(xx)//2):
        temp = xx[i]
        xx[i]=xx[-i - 1]
        xx[-i - 1]=temp

def is_palindrome(msg) :
    gsm = copy.deepcopy(msg)
    reverse(gsm)
    return msg == gsm

def trim_and_pad(xx, w) :
    return [(x+' '*w)[:w] for x in xx]

def wordle_difference(x, y) :
    assert len(x) == 5
    assert len(y) == 5
    xx = [c for c in x]
    yy = [c for c in y]
    same = sum([int(xn == yn) for xn, yn in zip(xx, yy)])
    diffX = (xn for xn, yn in zip(xx, yy) if xn != yn)
    diffY = (yn for xn, yn in zip(xx, yy) if xn != yn)
    xxx = set(diffX)
    yyy = set(diffY)
    diff = len(xxx.intersection(yyy))
    return (same, diff)
