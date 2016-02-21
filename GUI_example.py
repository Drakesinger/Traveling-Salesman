__author__ = 'Horia Mut, Mathieu Bandelier'

import pygame
from pygame.locals import KEYDOWN, QUIT, MOUSEBUTTONDOWN, K_RETURN, K_ESCAPE
import sys
import math
import random
from copy import deepcopy


# =====================================================
# "Algorithm"
# =====================================================

def ga_solve(file=None, gui=True, maxtime=0):
    cities = []

    g = None

    if file is None:
        g = GUI()
        cities = g.get_user_input()
    else:
        with open(file, 'r+') as f:
            for l in f.readlines():
                cities.append(l.split())

    # Load the problem and initialize.
    problem = Problem(cities)
    problem.initialize()

    print("todo")


# =====================================================
# Utilities
# =====================================================

def equal_double(a, b, epsilon=1e-6):
    """
    Returns true if a and b are equal, with epsilon as accepted range.
    False if not.
    """
    return abs(a - b) < epsilon


# =====================================================
# Data Parsing
# =====================================================

class City:
    PRECALCULATE_LIST = []

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.visited = False

    def __repr__(self):
        return self.id + "[" + self.x + " ; " + self.y + "]"

    def get_distance(self, other):
        """
        Get the euclidean distance between this city and another.
        :param other: Another city.
        :return: Euclidean distance between the two.
        """
        return math.hypot(self.x - other.x, self.y - other.y)

    @staticmethod
    def compute_all_possible_distances(cities):
        for k1 in range(0, len(cities)):
            City.PRECALCULATE_LIST.append(list())
            for k2 in range(0, len(cities)):
                City.PRECALCULATE_LIST[k1].append(0)

        for k1 in range(0, len(cities)):
            for k2 in range(k1, len(cities)):
                c1 = cities[k1].id
                c2 = cities[k2].id
                City.PRECALCULATE_LIST[c1][c2] = City.PRECALCULATE_LIST[c2][c1] = math.hypot(
                    cities[k1].x - cities[k2].x, cities[k1].y - cities[k2].y)


# class Solution:
# A solution to a problem is a trajectory passing once (and only once) through each city in P.
# AKA graph
# TODO

class Solution:
    """
    Class which represents a solution in the TSP. Each gene is represented by an unique id, related with the city.
    """

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.distance = 0

    def __repr__(self):
        return str(self.distance) + " : " + " ".join([str(i) for i in self.chromosome])

    def __len__(self):
        return len(self.chromosome)

    def __getitem__(self, item):
        return self.chromosome[item]

    def __setitem__(self, key, value):
        self.chromosome[key] = value

    def index(self, value):
        return self.chromosome.index(value)

    def compute_distance(self):
        """
        Computes for the traveling distance for one soltution.
        """
        self.distance = 0.0
        for s in xrange(0, len(self.chromosome) - 1):
            self.distance += City.compute_distance(self.chromosome[s], self.chromosome[s + 1])

        # do not forget to compute the distance between the first and the last city.
        self.distance += City.compute_distance(self.chromosome[0], self.chromosome[-1])

    def mutate(self):
        """
        Mutation of a solution where the path between two genes are inversed.
        i.e.: [0,1,2,3,4,5,6,7,8,9,10,11]  --> select random 5 and 8
              [0,1,2,3,4,8,7,6,5,9,10,11]
        """
        gene1 = random.randint(0, len(self.chromosome) - 1)
        gene2 = gene1
        while gene2 == gene1:
            gene2 = random.randint(0, len(self.chromosome) - 1)
        if gene1 > gene2:
            gene1, gene2 = gene2, gene1
        while gene1 < gene2:
            self.chromosome[gene1], self.chromosome[gene2] = self.chromosome[gene2], self.chromosome[gene1]
            gene1 += 1
            gene2 -= 1


# class Population:
# An array of solutions

class Problem:
    """
    Class that represents the problem.
    """

    NB_POPULATION = 0  # Will be changed during the execution time, by FACTOR*len(cities)
    FACTOR = 1
    SIZE_TOURNAMENT_BATTLE = 10  # Size of the tournament battle with which we keep the best
    MUTATION_RATE = 0.3  # Probability to mutate
    CROSSOVER_FRACTION = 0.8  # Number of generated offsprings
    DELTA_GENERATION = 50  # Convergence criteria. If the best solution hasn't changed since DELTA_GENERATION => STOP

    def __init__(self, filename=None, cities=None):
        """
        Initializes a problem, based on the cities passed as argument.
        The cities are supplied in format city objects.
        """
        Problem.NB_POPULATION = len(cities) * Problem.FACTOR
        self.cities_dict = {}
        self.keys = range(0, len(cities))
        self.best_solution = None
        self.population = []
        cities_id = []
        for city in cities:
            self.cities_dict[city.id] = city
            cities_id.append(city.id)
        City.compute_all_possible_distances(cities_id)

    def print_problem(self):
        for city_key in self.cities_dict.keys():
            print(self.cities_dict.get(city_key))

    def create_population(self):
        """
        Creates a population based on the keys passed as argument.
        Returns the population.
        """
        for i in xrange(0, Problem.NB_POPULATION):
            random.shuffle(self.keys)  # Use Fisher-Yates shuffle, O(n). Better than copying and removing
            self.population.append(Solution(self.keys[:]))

    def initialize(self):
        """
        Preparation for the execution of the algorithm.
        """
        self.best_solution = Solution([])
        self.best_solution.distance = float('inf')
        self.create_population()
        self.compute_all_distances()

    def compute_all_distances(self):
        """
        Computes the distances for all the solutions availlable in the population.
        Determines also the best_solution in the population.
        """
        for p in self.population:
            p.compute_distance()
            if p.distance < self.best_solution.distance and not equal_double(p.distance, self.best_solution.distance):
                self.best_solution = deepcopy(p)

    def generate(self):
        """
        Runs all the steps for the generation of a "good" solution.
        Returns the best solution obtained during the generation.
        """
        new_population = self.selection_process()
        new_population += self.crossover_process(new_population)
        self.mutation_process(new_population)

        self.population = new_population
        self.compute_all_distances()

        # If we don't have enough cities to realize a crossover (eg 5)
        if len(self.population) > Problem.NB_POPULATION:
            self.population.sort(key=lambda p: p.distance)
            self.population = self.population[:Problem.NB_POPULATION]
        return self.best_solution

    def selection_process(self):
        """
        Runs the tournament with a specified size (defined as static).
        """
        new_population = []
        # If the number of cities is to small, we return the entire population and we'll cut it later
        if self.SIZE_TOURNAMENT_BATTLE >= len(self.population):
            return self.population
        else:
            for i in xrange(0, int(round((1 - Problem.CROSSOVER_FRACTION) * Problem.NB_POPULATION))):
                solutions = random.sample(self.population, self.SIZE_TOURNAMENT_BATTLE)
                solutions.sort(key=lambda p: p.distance)
                self.population.remove(solutions[
                                           0])  # O(n) but if we want, we could do the tricks with swaping with the last element and then pop it. But the population is really small so not necessary
                new_population.append(solutions[0])
        return new_population

    def crossover_process(self, new_population):
        """
        Does the crossover of two random solutions
        """
        future_solution = []
        for i in xrange(0, int(round(Problem.NB_POPULATION * Problem.CROSSOVER_FRACTION) / 2)):
            solution1 = random.choice(new_population)
            solution2 = solution1
            while solution2 == solution1:
                solution2 = random.choice(new_population)

            future_solution.append(self.crossover(solution1, solution2))
            future_solution.append(self.crossover(solution2, solution1))
        return future_solution

    def crossover(self, ga, gb):
        fa, fb = True, True
        n = len(ga)
        city = random.choice(ga.chromosome)
        x = ga.index(city)
        y = gb.index(city)
        g = [city]

        while fa or fb:
            x = (x - 1) % n
            y = (y + 1) % n
            if fa:
                if ga[x] not in g:
                    g.insert(0, ga[x])
                else:
                    fa = False
            if fb:
                if gb[y] not in g:
                    g.append(gb[y])
                else:
                    fb = False

        remaining_cities = []
        if len(g) < len(ga):
            while len(g) + len(remaining_cities) != n:
                x = (x - 1) % n
                if ga[x] not in g:
                    remaining_cities.append(ga[x])
            random.shuffle(remaining_cities)  # Use Fisher-Yates shuffle, O(n). Better than copying and removing
            while len(remaining_cities) > 0:
                g.append(remaining_cities.pop())

        return Solution(g)

    def mutation_process(self, new_population):
        """
        Mutates some of the solutions in the new_population passed as argument.
        """
        for s in random.sample(new_population, int(round(Problem.MUTATION_RATE * Problem.NB_POPULATION))):
            s.mutate()


# =====================================================
# Pygame Window
# =====================================================

import pygame
from pygame.locals import KEYDOWN, QUIT, MOUSEBUTTONDOWN, K_RETURN, K_ESCAPE
import sys


class GUI:
    screen_x = 500
    screen_y = 500

    city_color = [10, 10, 200]  # blue
    city_radius = 3

    font_color = [255, 255, 255]  # white

    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((GUI.screen_x, GUI.screen_y))
        pygame.display.set_caption('Exemple')
        self.screen = pygame.display.get_surface()
        self.font = pygame.font.Font(None, 30)

        self.cities = []
        self.paths_to_draw = []

    def refresh(self):
        self.screen.fill(0)
        for city in self.cities:
            pygame.draw.circle(self.screen, self.city_color, (int(city.x), int(city.y)), self.city_radius)
        text = self.font.render("Nombre: %i" % len(self.cities), True, self.font_color)
        textRect = text.get_rect()
        self.screen.blit(text, textRect)
        pygame.display.flip()

    def get_user_input(self):
        self.refresh()
        collecting = True

        while collecting:
            for event in pygame.event.get():
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN and event.key == K_RETURN:
                    collecting = False
                elif event.type == MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    city = City("City" + str(len(self.cities)), x, y)
                    self.cities.append(city)
                    self.refresh()
        return self.cities

    def get_path(self):
        self.screen.fill(0)
        # Draw the path here.
        pygame.draw.lines(self.screen, self.city_color, True, self.cities)
        text = self.font.render("Un chemin, pas le meilleur!", True, self.font_color)
        textRect = text.get_rect()
        self.screen.blit(text, textRect)
        pygame.display.flip()


# while True:
#     event = pygame.event.wait()
#     if event.type == KEYDOWN: break

def handle_argv():
    """
    usage: MutBandelier.py [options] [parameters]

    options:
    -n, --no-gui                Disable graphical user interface.

    parameters:
    -m VALUE, --maxtime=VALUE   Maximum execution time.
    -f VALUE, --filename=VALUE  File containing city coords.

    (c) 2016 by Horia Mut and Mathieu Bandelier."""

    import getopt
    opts = []
    try:
        opts = getopt.getopt(
            sys.argv[1:],
            "nm:f:",
            ["no-gui", "maxtime=", "filename="])[0]
    except getopt.GetoptError:
        print(handle_argv.__doc__)
        sys.exit(2)

    show_gui = True
    max_time = 0
    filename = None

    for opt, arg in opts:
        if opt in ("-n", "--no-gui"):
            show_gui = False
        elif opt in ("-m", "--maxtime"):
            max_time = int(arg)
        elif opt in ("-f", "--filename"):
            filename = str(arg)

    return filename, show_gui, max_time


# =====================================================
# Main Function
# =====================================================
if __name__ == '__main__':
    filename, show_gui, max_time = handle_argv()

    ga_solve(filename, show_gui, max_time)
