# coding: latin-1

"""Un script quick-and-dirty pour générer des problèmes du voyageur de commerce.

Usage: generate_cities <nombre> <fichier>

Va générer <nombre> couple de nombres et le mettre dans <fichier> au format
v1 x1 y2
v2 x2 y2
...

Attention! script sans garantie! notamment, si <fichier> existe, IL SERA ECRASÉ!!!

""" 


import sys
from random import randint

MAX_X = MAX_Y = 500

try:
    filename = sys.argv[2]
    nb = int(sys.argv[1])
except:
    print (__doc__)
    sys.exit(1)

f = open(filename, "w")

for i in range(nb):
    line = "v%d %d %d\n" % (i, randint(0,MAX_X), randint(0,MAX_Y))
    f.write(line)

f.close()