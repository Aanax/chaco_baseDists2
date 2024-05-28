import sys
import os

def isInt(value):
    try:
        int(value)
        return True
    except ValueError:
        return False



with open(sys.argv[1],'r') as f:
    for line in f:
        for word in line.split():
            if isInt(word):
                os.system('scancel %d' %int(word))
            