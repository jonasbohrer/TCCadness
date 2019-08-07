class Modules:
    """
    Represents a set of one or more basic units of a topology.
    components: the most basic unit of a topology.
    """
    def __init__(self):
        self.components = None

class Genome:
    """
    Represents a topology made of modules.
    modules: modules composing a topology.
    """
    def __init__(self):
        self.modules = None

class Species:
    """
    Represents a group of topologies with similarities.
    property: a set of common properties defining a species.
    """
    def __init__(self):
        self.property = None

class Individual:
    """
    Represents a topology in the Genetic Algorithm scope.
    genome: representation of the topology.
    species: a grouping of topologies set according to similarity.
    birth: epoch the genome was created.
    parents: individuals used in crossover to generate this topology.
    """
    def __init__(self):
        self.genome = None
        self.species = None
        self.birth = None
        self.parents = None

    def generate(self):
        """
        Returns the keras model representing of the topology.
        """
        pass

    def fit(self):
        """
        Fits the keras model representing of the topology.
        """
        pass

class Population:
    """
    Represents the population containing multiple individual topologies and their correlations.
    """
    def __init__(self):
        self.members = []
        self.species = []
        self.groups = []

    def create(self):
        """
        Creates a random population of individuals.
        """
        pass
    
    def iterate(self):
        """
        Fits the individuals and generates scores.
        """
        pass

    def speciate(self):
        """
        Divides the individuals in groups according to similarity.
        """
        pass

    def evaluate(self):
        """
        Choses which individuals will be kept in the population.
        """
        pass

    def crossover(self):
        """
        Generates new individuals based on existing individuals.
        """
        pass
