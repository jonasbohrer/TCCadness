"""
Genome example:

keras.models.Sequential([
            keras.layers.Conv2D(48, (3, 3), activation="relu", input_shape=input_shape),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Conv2D(24, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dense(16, activation="softmax"),
            ])

"""

import keras, logging
from enum import Enum, auto

class LayerPosition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    OUTPUT = "output"
    COMPILER = "compiler"

class Datasets:
    def __init__(self, complete=None, test=None, training=None, validation=None):
        self.complete=complete
        self.training=training
        self.validation=validation
        self.test=test
    
    @property
    def shape(self):
        return self.complete.shape

    def split_complete(self):
        """
        Returns a full split from the complete dataset into training and validation
        """
        pass

class Component:
    """
    Represents a basic unit of a topology.
    keras_component: the keras component being represented
    """
    def __init__(self, representation=None, keras_component=None):
        self.representation = representation
        self.keras_component = keras_component

class Module:
    """
    Represents a set of one or more basic units of a topology.
    components: the most basic unit of a topology.
    """
    def __init__(self, components=None, layer_type=LayerPosition.INPUT):
        self.components = components
        self.layer_type = layer_type
    
    def __getitem__(self, item):
        return self.components[item]

class Genome:
    """
    Represents a topology made of modules.
    modules: modules composing a topology.
    """
    def __init__(self, modules=None, compiler=None):
        self.modules = modules
        self.compiler = compiler

    def __getitem__(self, item):
        return self.modules[item]

class Species:
    """
    Represents a group of topologies with similarities.
    property: a set of common properties defining a species.
    """
    def __init__(self, group=None):
        self.group = group

class Individual:
    """
    Represents a topology in the Genetic Algorithm scope.
    genome: representation of the topology.
    species: a grouping of topologies set according to similarity.
    birth: epoch the genome was created.
    parents: individuals used in crossover to generate this topology.
    """
    def __init__(self, genome=None, species=None, birth=None, parents=None, model=None):
        self.genome = genome
        self.species = species
        self.birth = birth
        self.parents = parents
        self.model = None

    @staticmethod
    def generate(self):
        """
        Returns the keras model representing of the topology.
        """

        logging.info(f"Generating keras layers")
        keras_topology = []

        for module in self.genome:
            if module.layer_type == LayerPosition.OUTPUT:
                logging.info(f"Adding a Flatten() layer")
                keras_topology.append(keras.layers.Flatten())
            for component in module:
                keras_topology.append(component.keras_component)
        
        self.model = keras.models.Sequential(keras_topology)
        self.model.compile(**self.genome.compiler)

    def fit(self, input_x, input_y, epochs=1):
        """
        Fits the keras model representing the topology.
        """

        logging.info(f"Fitting one individual for {epochs} epochs")

        self.generate()
        fitness = self.model.fit(input_x, input_y, epochs=epochs)

        return fitness

    def extract_genome(self):
        """
        Generates a Genome description of the self.model variable.
        """
        pass

class Population:
    """
    Represents the population containing multiple individual topologies and their correlations.
    """
    def __init__(self, datasets=None, individuals=[], species=[], groups=[]):
        self.datasets = datasets
        self.individuals = individuals
        self.species = species
        self.groups = groups

    def create(self, size: int =1):
        """
        Creates a random population of individuals.
        """
        pass
    
    def iterate_fitness(self):
        """
        Fits the individuals and generates scores.
        """

        logging.info(f"Iterating fitness over {len(self.individuals)} individuals")
        iteration = []

        #(batch, channels, rows, cols)
        input_x = self.datasets.training[0]
        input_y = self.datasets.training[1]

        for individual in self.individuals:
            score = individual.fit(input_x, input_y)
            iteration.append(score)

        return iteration

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

    def iterate_epochs(self, epochs=1):
        """
        Manages epoch iterations, applying the genetic algorithm in fact.
        """

        logging.info(f"Iterating over {epochs} epochs")
        iterations = []
        for epoch in range(epochs):
            iterations.append(self.iterate_fitness())

        return iterations

def test_run():
    """
    A test run to verify the whole pipeline.
    """

    class MyPopulation(Population):
        """
        An example child class of Population class.
        """

        def create(self, size=1):
        
            new_individuals = []

            for n in range(size):
                #Create components
                new_input_conv_component = Component(None, keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=(8, 8, 1)))
                new_conv_component = Component(None, keras.layers.Conv2D(4, (3, 3), activation="relu"),)
                new_softmax_component = Component(None, keras.layers.Dense(2, activation="softmax"),)
                new_compiler = {"loss":"sparse_categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}
                #Combine them in modules
                new_input_module = Module([new_input_conv_component], LayerPosition.INPUT)
                new_intermed_module = Module([new_conv_component], LayerPosition.INTERMED)
                new_output_module = Module([new_softmax_component], LayerPosition.OUTPUT)
                #Create the genome combining modules
                new_genome = Genome([new_input_module, new_intermed_module, new_output_module], new_compiler)
                #Create individual with the Genome
                new_individual = Individual(genome=new_genome)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

    training_dataset = [[[[[[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]]],
                                 [[[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]]],
                                 [[[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]]],
                                 [[[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]]],
                                 [[[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]]],
                                 [[[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[0], [0], [0], [0], [0], [0], [0], [0]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]],
                                  [[1], [1], [1], [1], [1], [1], [1], [1]]]]], [1, 0, 1, 0, 1, 0]]

    my_dataset = Datasets(training=training_dataset)

    logging.basicConfig(filename='test.log',
                        filemode='w+', level=logging.DEBUG,
                        format='%(levelname)s - %(asctime)s: %(message)s')

    logging.warning('This will get logged to a file')
    logging.info(f"Hi, this is a test run.")

    population = MyPopulation(my_dataset)
    population.create(5)

    iteration = population.iterate_epochs(5)


if __name__ == "__main__":
    
    test_run()