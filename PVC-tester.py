# coding: latin-1

''' Module permettant de tester systématiquement une série de solveurs 
pour le problème du voyageur de commerce.

Permet de lancer automatiquement une série de solveurs sur une série de problèmes
et génère une grille de résultats au format CSV.

v0.2, Matthieu Amiguet, HE-Arc
v0.3, hatem Ghorbel, HE-Arc
'''

# PARAMETRES
# =========
# modifier cette partie pour l'adapter à vos besoins

# Le nom des modules à tester
# Ces modules doivent être dans le PYTHONPATH; p.ex. dans le répertoire courant

modules = (
	"MAA09",
	# Éventuellement d'autres modules pour comparer plusieurs versions...
)

# Liste des tests à effectuer 
# sous forme de couples (<datafile>, <maxtime>) où
# <datafile> est le fichier contenant les données du problème et
# <maxtime> le temps (en secondes) imparti pour la résolution
tests = (
    ('data/pb005.txt',1),
    #~ ('data/pb010.txt',5),
    #~ ('data/pb010.txt',10),
    #~ ('data/pb050.txt',30),
    #~ ('data/pb050.txt',60),
    #~ ('data/pb100.txt',20),
    #~ ('data/pb100.txt',90),
)

# On tolère un dépassement de 5% du temps imparti:
tolerance = 0.05

# Fichier dans lequel écrire les résultats
import sys
outfile = sys.stdout
# ou :
#outfile = open('results.csv', 'w')

# affichage à la console d'informations d'avancement?
verbose = False

# est-ce qu'on veut un affichage graphique?
gui = False

# PROGRAMME
# =========
# Cette partie n'a théoriquement pas à être modifiée

import os
from time import time
from math import hypot

def dist(x1,y1,x2,y2):
    return hypot(x2 -x1,y2-y1)

def validate(filename, length, path, duration, maxtime):
    '''Validation de la solution
    
    retourne une chaîne vide si tout est OK ou un message d'erreur sinon
    '''
    error = ""
    
    if duration>maxtime * (1+tolerance):
        error += "Timeout (%.2f) " % (duration-maxtime)
    try:
        cities = dict([(name, (int(x),int(y))) for name,x,y in [l.split() for l in file(filename)]])
    except:
        return "(Validation failed...)"
    tovisit = cities.keys()
    
    try:
        totaldist = 0
        for (ci, cj) in zip(path, path[1:] +path[0:1]):
            totaldist += dist(cities[ci],cities[cj])
            tovisit.remove(ci)
            
        if int(totaldist) != int(length):
            error += "Wrong dist! (%d instead of %d)" % (length, totaldist)
    except KeyError:
        error += "City %s does not exist! " % ci
    except ValueError:
        error += "City %s appears twice in %r! " % (ci, path)
    except Exception as e:
        error += "Error during validation: %r" % e
    
    if tovisit:
        error += "Not all cities visited! %r" % tovisit
    
    return error



if __name__ == '__main__':
    # Récupération des différentes implémentations
    # On met les différentes fonctions ga_solve() dans un dictionnaire indexé par le nom du module correpsondant
    # On en profite pour écrire la ligne d'en-tête du fichier de sortie

    solvers = {}

    outfile.write('Test;')

    for m in modules:
        exec ("from %s import ga_solve" % m)
        solvers[m] = ga_solve
        outfile.write("%s;" % m)

    outfile.write('\n')

    # Cette partie effectue les tests proprement dits
    # et rapporte les résultats dans outfile

    for (filename, maxtime) in tests:
        if verbose: 
            print ("--> %s, %d" % (filename, maxtime))
        # normalisation du nom de fichier (pour l'aspect multi-plateforme)
        filename = os.path.normcase(os.path.normpath(filename))
        # Écriture de l'en-tête de ligne
        outfile.write("%s (%ds);" % (filename, maxtime))
        # Appel des solveurs proprement dits, vérification et écriture des résultats
        for m in modules:
            if verbose: 
                print ("## %s" % m)
            try:
                start = time()
                length, path = solvers[m](filename, gui, maxtime)
                duration = time()-start
            except Exception as e:
                    outfile.write("%r;" % e)
            except SystemExit:
                outfile.write("tried to quit!;")
            else:
                error = validate(filename, length, path, duration, maxtime)
                if not error:
                    outfile.write("%d;" % length)
                else:
                    outfile.write("%s;" % error)
            outfile.flush()
        outfile.write('\n')