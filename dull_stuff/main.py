"""
Blueprint example:

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

import keras, logging, random, pydot
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List
from keras.utils.vis_utils import plot_model

input_configs = {
    "module_range" : ([1, 1], 'int'),
    "component_range" : ([1, 1], 'int')
}

global_configs = {
    "module_range" : ([2, 4], 'int'),
    "component_range" : ([4, 6], 'int')
}

output_configs = {
    "module_range" : ([1, 1], 'int'),
    "component_range" : ([1, 1], 'int')
}

possible_inputs = {
    "conv2d": (keras.layers.Conv2D, {"filters": ([48,48], 'int'), "kernel_size": ([3], 'list'), "activation": (["relu"], 'list')})
}

possible_components = {
    #"conv2d": (keras.layers.Conv2D, {"filters": ([8, 48], 'int'), "kernel_size": ([1], 'list'), "strides": ([1], 'list'), "data_format": (['channels_first'], 'list'), "padding": (['same'], 'list')}),
    "dense": (keras.layers.Dense, {"units": ([8, 48], 'int')})
}

possible_outputs = {
    "dense": (keras.layers.Dense, {"units": ([2,2], 'int'), "activation": (["softmax"], 'list')})
}

possible_complementary_components = {
    "maxpooling2d": (keras.layers.MaxPooling2D, {"pool_size": ([2, 3], 'list')}),
    "dropout": (keras.layers.Dropout, {"rate": ([0, 0.5], 'float')})
}

class HistoricalMarking:
    def __init__(self):
        counter = 0
    
    def generate(self):
        counter += 1
        return counter

class ModulePosition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    OUTPUT = "output"
    COMPILER = "compiler"

class ComponentParameters(Enum):
    CONV2D = (keras.layers.Conv2D, {"filters": ([8, 16], 'int'), "kernel": ([3, 5, 7], 'list'), "stride": ([1, 2, 3], 'list')})
    MAXPOOLING2D = (keras.layers.MaxPooling2D, {"kernel":[3, 5, 7]})
    FLATTEN = (keras.layers.Flatten, 0)
    DENSE = (keras.layers.Dense, {"units":128, "activation":"relu"})

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

class Component(object):
    """
    Represents a basic unit of a topology.
    keras_component: the keras component being represented
    """
    def __init__(self, representation, keras_component=None):
        self.representation = representation
        self.keras_component = keras_component
        self.historical_marking = 0

class Module(object):
    """
    Represents a set of one or more basic units of a topology.
    components: the most basic unit of a topology.
    """
    def __init__(self, components:dict, layer_type:ModulePosition=ModulePosition.INPUT, component_graph=None):
        self.components = components
        self.component_graph = component_graph
        self.layer_type = layer_type
        self.species = None
    
    def __getitem__(self, item):
        return self.components[item]
    
    def compare_to(self, individual):
        pass

class Blueprint:
    """
    Represents a topology made of modules.
    modules: modules composing a topology.
    """
    def __init__(self, modules:List[Module], compiler=None, input_shape=None, blueprint_graph=None):
        self.modules = modules
        self.compiler = compiler
        self.input_shape = input_shape
        self.blueprint_graph = blueprint_graph

    def __getitem__(self, item):
        return self.modules[item]

class Species:
    """
    Represents a group of topologies with similarities.
    properties: a set of common properties defining a species.
    """
    def __init__(self, group=None, properties=None):
        self.group = group
        self.properties = properties

class Individual:
    """
    Represents a topology in the Genetic Algorithm scope.
    blueprint: representation of the topology.
    species: a grouping of topologies set according to similarity.
    birth: epoch the blueprint was created.
    parents: individuals used in crossover to generate this topology.
    """
       
    def __init__(self, blueprint:Blueprint, birth=None, parents=None, model=None):
        self.blueprint = blueprint
        self.birth = birth
        self.parents = parents
        self.model = None

    def generate_old(self):
        """
        Returns the keras model representing of the topology.
        """

        logging.info(f"Generating keras layers")
        module_map = {}
        module_level = 0

        for module in self.blueprint:
            first_layer = None
            last_layer = None
            layer_map = {}
            layer_level = 0
                
            for component_map in module:
                logging.info(f"Module level: {module_level}; Layer level: {layer_level}")
                input_connection, output_connection, component = component_map

                """if output_connection == None:
                    preceding_component = component.keras_component"""

                if input_connection == None:
                    #if module.layer_type == ModulePosition.INPUT:
                    if module_level == 0 and layer_level == 0:
                        logging.info(f"Adding the Input() layer")
                        model_input = keras.layers.Input(shape=self.blueprint.input_shape)
                        logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {model_input}")
                        first_layer = component.keras_component(model_input)
                        logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {first_layer}")
                    else:
                        logging.info(f"Adding first layer")
                        preceding_layer = module_map[module_level-1][max(module_map[module_level-1])]
                        #print(preceding_layer, (isinstance(preceding_component, keras.layers.Conv2D) or "conv2d" in preceding_layer.name), component.keras_component, isinstance(component.keras_component, keras.layers.Dense))
                        #print(preceding_layer.name, preceding_component.name, "conv2d" in preceding_component.name)
                        if ("conv2d" in preceding_layer.name) and isinstance(component.keras_component, keras.layers.Dense):
                            logging.info(f"Adding a Flatten() layer")
                            #preciding_layer = keras.layers.MaxPool2D((2,2))
                            preceding_layer = keras.layers.Flatten()(preceding_layer)
                            logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {preceding_layer}")
                        first_layer = component.keras_component(preceding_layer)
                        logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {first_layer}")
                    layer_map[layer_level] = first_layer
                elif isinstance(input_connection, tuple):
                    logging.info(f"Adding a Merge layer")
                    merge_layer = keras.layers.concatenate([layer_map[layer] for layer in input_connection])
                    logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {merge_layer}")
                    intermed_layer = component.keras_component(merge_layer)
                    logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {intermed_layer}")
                    layer_map[layer_level] = intermed_layer
                else:
                    print(layer_map[input_connection], isinstance(layer_map[input_connection], keras.layers.Conv2D), "conv2d" in layer_map[input_connection].name)
                    preceding_layer = layer_map[input_connection]
                    if (isinstance(layer_map[input_connection], keras.layers.Conv2D) or "conv2d" in layer_map[input_connection].name) and isinstance(component.keras_component, keras.layers.Dense):
                        logging.info(f"Adding a Flatten() layer")
                        preceding_layer = keras.layers.Flatten()(preceding_layer)
                        logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {preceding_layer}")
                    logging.info(f"Adding the Intermediate layer")
                    intermed_layer = component.keras_component(layer_map[input_connection])
                    logging.log(21, f"{layer_level}: {input_connection}, {output_connection} - Added {intermed_layer}")
                    layer_map[layer_level] = intermed_layer
                layer_level += 1
                module_map[module_level] = layer_map

                preceding_component = component.keras_component
                
            module_map[module_level] = layer_map
            module_level += 1

        print(module_map)
        logging.log(21, module_map)
        self.model = keras.models.Model(inputs=model_input, outputs=module_map[module_level-1][layer_level-1])
        self.model.compile(**self.blueprint.compiler)

    def generate(self):
        """
        Returns the keras model representing of the topology.
        """

        layer_map = {}
        blueprint_graph = self.blueprint.blueprint_graph

        assembled_blueprint_graph = nx.DiGraph()
        output_nodes = []

        for node in blueprint_graph.nodes():
            assembled_blueprint_graph = nx.union(assembled_blueprint_graph, blueprint_graph.nodes[node]["module_def"].component_graph, rename=(None, f'{node}-'))
            output_nodes.append(max(blueprint_graph.nodes[node]["module_def"].component_graph.nodes()))

        for node in blueprint_graph.nodes():
            for successor in blueprint_graph.successors(node):
                assembled_blueprint_graph.add_edge(f'{node}-{output_nodes[node]}', f'{successor}-0')

        plt.subplot(121)
        nx.draw(assembled_blueprint_graph, nx.drawing.nx_agraph.graphviz_layout(assembled_blueprint_graph, prog='dot'), with_labels=True, font_weight='bold')
        plt.savefig("Graph.png", format="PNG")

        logging.log(21, f"Generated the assembled graph")

        logging.info(f"Generating keras layers")
        logging.info(f"Adding the Input() layer")
        model_input = keras.layers.Input(shape=self.blueprint.input_shape)
        logging.log(21, f"Added {model_input}")

        component_graph = assembled_blueprint_graph

        for component_id in component_graph.nodes():
            component = component_graph.nodes[component_id]["component_def"]

            if component_graph.in_degree(component_id) == 0:
                layer = component.keras_component(model_input)
                logging.log(21, f"{component_id}: Added {layer}")
            
            elif component_graph.in_degree(component_id) == 1:
                predecessors = [layer_map[predecessor_id] for predecessor_id in component_graph.predecessors(component_id)][0]
                print("conv2d" in predecessors.name, "dense" in component.keras_component.name)
                if "conv2d" in predecessors.name and "dense" in component.keras_component.name:
                    logging.info(f"Adding a Flatten() layer")
                    layer = keras.layers.Flatten()(predecessors)
                    logging.log(21, f"{component_id}: Added {layer}")
                    predecessors = layer
                layer = component.keras_component(predecessors)
                logging.log(21, f"{component_id}: Added {layer}")

            elif component_graph.in_degree(component_id) == 2:
                predecessors = [layer_map[predecessor_id] for predecessor_id in component_graph.predecessors(component_id)]

                logging.info(f"Adding a Merge layer")
                merge_layer = keras.layers.concatenate(predecessors)
                logging.log(21, f"{component_id}: Added {merge_layer}")
                layer = component.keras_component(merge_layer)
                logging.log(21, f"{component_id}: Added {layer}")

            layer_map[component_id] = layer

        logging.log(21, layer_map)
        self.model = keras.models.Model(inputs=model_input, outputs=layer_map[max(layer_map)])
        self.model.compile(**self.blueprint.compiler)
        plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    def fit(self, input_x, input_y, epochs=1):
        """
        Fits the keras model representing the topology.
        """

        logging.info(f"Fitting one individual for {epochs} epochs")

        self.generate()
        fitness = self.model.fit(input_x, input_y, epochs=epochs)

        return fitness

    def extract_blueprint(self):
        """
        Generates a Blueprint description of the self.model variable.
        """
        pass

class Population:
    """
    Represents the population containing multiple individual topologies and their correlations.
    """
    def __init__(self, datasets=None, individuals=[], modules=[], hyperparameters=[], species=[], groups=[]):
        self.datasets = datasets
        self.individuals = individuals
        self.modules = modules
        self.hyperparameters = hyperparameters
        self.species = species
        self.groups = groups

    def create(self, size: int =1):
        """
        Creates a specific population of individuals.
        """
        pass

    def create_random(self, size: int =1):
        """
        Creates a random population of individuals.
        """
        pass

    def iterate_fitness(self, training_epochs=1):
        """
        Fits the individuals and generates scores.
        """

        logging.info(f"Iterating fitness over {len(self.individuals)} individuals")
        iteration = []

        #(batch, channels, rows, cols)
        input_x = self.datasets.training[0]
        input_y = self.datasets.training[1]

        for individual in self.individuals:
            score = individual.fit(input_x, input_y, training_epochs)
            iteration.append(score)

        return iteration

    def speciate(self):
        """
        Divides the individuals in groups according to similarity.


        score = c2*(parameter similarity in matching components)/(component amount)
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

    def iterate_epochs(self, epochs=1, training_epochs=1):
        """
        Manages epoch iterations, applying the genetic algorithm in fact.
        """

        logging.info(f"Iterating over {epochs} epochs")
        iterations = []
        for epoch in range(epochs):
            iterations.append(self.iterate_fitness(training_epochs))

        return iterations

def test_run(global_configs, possible_components):
    """
    A test run to verify the whole pipeline.
    """
    class Generator():

        @staticmethod
        def random_parameter_def(possible_parameters, parameter_name):
            parameter_def = {}
            parameter_values, parameter_type = possible_parameters[parameter_name]
            if parameter_type == 'int':
                parameter_def = random.randint(parameter_values[0], parameter_values[1])
            elif parameter_type == 'float':
                parameter_def = ((parameter_values[1] - parameter_values[0]) * random.random()) + parameter_values[0]
            elif parameter_type == 'list':
                parameter_def = random.choice(parameter_values)

            return parameter_def

        def random_component(self, possible_components):
            component_def, possible_parameters = possible_components[random.choice(list(possible_components))]
            print(possible_parameters)

            parameter_def = {}
            for parameter_name in possible_parameters:
                parameter_def[parameter_name] = self.random_parameter_def(possible_parameters, parameter_name)
            print(parameter_def)

            new_component = Component([component_def, parameter_def], component_def(**parameter_def))
            return new_component

        def random_module(self, global_configs, possible_components):

            component_graph = nx.DiGraph()
            components = []
            
            component_range = self.random_parameter_def(global_configs, "component_range")
            print("global parameter def", component_range)

            for component in range(component_range):
                component_def = self.random_component(possible_components)

                component_graph.add_node(component, component_def=component_def)
                if component == 0:
                    input_connection = None
                    output_connection = None
                elif component > 0 and component < component_range-1:
                    precedent = random.randint(0, component-1)
                    component_graph.add_edge(precedent, component)

                    input_connection = precedent
                    output_connection = None
                elif component == component_range-1:
                    leaf_nodes = [node for node in component_graph.nodes() if component_graph.out_degree(node)==0]
                    leaf_nodes.remove(component)
                    while (len(leaf_nodes) > 0):
                        if len(leaf_nodes) <= 2:
                            for node in leaf_nodes:
                                component_graph.add_edge(node, component)

                            input_connection = tuple(leaf_nodes)
                            output_connection = None
                        else:
                            random_node1 = random.choice(leaf_nodes)
                            leaf_nodes.remove(random_node1)

                            random_node2 = random.choice(leaf_nodes)
                            while (component_graph.in_degree(random_node2)>1):
                                random_node2 = random.choice(leaf_nodes)
                            component_graph.add_edge(random_node1, random_node2)
                        leaf_nodes = [node for node in component_graph.nodes() if component_graph.out_degree(node)==0]
                        leaf_nodes.remove(component)
                    
                components.append([input_connection, output_connection, component_def])
            
            """print(component_graph.nodes())
            print("components def", components)
            plt.subplot(121)
            nx.draw(component_graph, nx.drawing.nx_agraph.graphviz_layout(component_graph, prog='dot'), with_labels=True, font_weight='bold')
            plt.show()
            exit(0)"""
            new_module = Module(components, ModulePosition.INTERMED, component_graph=component_graph)

            return new_module


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
                new_input_module = Module([[None, None, new_input_conv_component]], ModulePosition.INPUT)
                new_intermed_module = Module([[None, None, new_conv_component]], ModulePosition.INTERMED)
                new_output_module = Module([[None, None, new_softmax_component]], ModulePosition.OUTPUT)
                #Create the blueprint combining modules
                input_shape = (8, 8, 1)
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], new_compiler, input_shape)
                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

        def create_random_straight(self, size=1):

            new_individuals = []

            for n in range(size):
                #Create input module
                new_input_conv_component = Component(None, keras.layers.Conv2D(48, (3, 3), activation="relu"))
                new_input_module = Module([[None, None, new_input_conv_component]], ModulePosition.INPUT)
                
                def new_module():
                    
                    """    parameter_values, parameter_type = possible_parameters[parameter]
                        if parameter_type == 'int':
                            parameter_def[parameter] = random.randint(parameter_values[0], parameter_values[1])
                        if parameter_type == 'float':
                            parameter_def[parameter] = ((parameter_values[1] - parameter_values[0]) * random.random()) + parameter_values[0]
                        if parameter_type == 'list':
                            parameter_def[parameter] = random.choice(parameter_values)"""
                    components = []
                    parameter_values, parameter_type = global_configs["component_range"]
                    components = random.randint(parameter_values[0], parameter_values[1])
                    new_module = Module([[None, 1, self.random_component()],
                                                    [0, 2, self.random_component()],
                                                    [1, 3, self.random_component()],
                                                    [2, None, self.random_component()]], ModulePosition.INPUT)
                    return new_module

                #new_input_module = new_module()
                #Create intermediate module
                new_intermed_module = Module([[None, 1, self.random_component()],
                                                [0, 2, self.random_component()],
                                                [1, 3, self.random_component()],
                                                [2, None, self.random_component()]], ModulePosition.INTERMED)

                #Create output module
                new_softmax_component = Component(None, keras.layers.Dense(2, activation="softmax"),)
                new_output_module = Module([[None, None, new_softmax_component]], ModulePosition.OUTPUT)

                #Set the compiler
                new_compiler = {"loss":"sparse_categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}
                input_shape = (8, 8, 1)
                #Create the blueprint combining modules
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], new_compiler, input_shape)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

        def create_random_forked(self, size=1):

            new_individuals = []

            for n in range(size):
                #Create input module
                new_input_conv_component = Component(None, keras.layers.Conv2D(48, (3, 3), activation="relu"))
                new_input_module = Module([[None, None, new_input_conv_component]], ModulePosition.INPUT)
              
                #Create intermediate module
                new_intermed_module = Module([[None, 1, self.random_component()],
                                                [0, 3, self.random_component()],
                                                [0, 3, self.random_component()],
                                                [(1,2), None, self.random_component()]], ModulePosition.INTERMED)

                #Create output module
                new_softmax_component = Component(None, keras.layers.Dense(2, activation="softmax"),)
                new_output_module = Module([[None, None, new_softmax_component]], ModulePosition.OUTPUT)

                #Set the compiler
                new_compiler = {"loss":"sparse_categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}
                input_shape = (8, 8, 1)
                #Create the blueprint combining modules
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], new_compiler, input_shape)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

        def create_random_modules(self, size=1):

            new_individuals = []

            for n in range(size):
                #Create input module
                new_input_module = Generator().random_module(input_configs, possible_inputs)
              
                #Create intermediate module
                new_intermed_module1 = Generator().random_module(global_configs, possible_components)
                new_intermed_module2 = Generator().random_module(global_configs, possible_components)

                #Create output module
                new_output_module = Generator().random_module(output_configs, possible_outputs)

                #Set the compiler
                new_compiler = {"loss":"sparse_categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}
                input_shape = (8, 8, 1)
                #Create the blueprint combining modules
                blueprint_graph = nx.DiGraph()
                
                blueprint_graph.add_node(0, module_def=new_input_module)
                blueprint_graph.add_node(1, module_def=new_intermed_module1)
                blueprint_graph.add_node(2, module_def=new_intermed_module2)
                blueprint_graph.add_node(3, module_def=new_output_module)
                blueprint_graph.add_edge(0, 1)
                blueprint_graph.add_edge(0, 2)
                blueprint_graph.add_edge(1, 3)
                blueprint_graph.add_edge(2, 3)

                new_blueprint = Blueprint([new_input_module, new_intermed_module1, new_intermed_module2, new_output_module], new_compiler, input_shape, blueprint_graph=blueprint_graph)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

    #A random distribution of 0s and 1s.
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
    logging.addLevelName(21, "TOPOLOGY")

    logging.warning('This will get logged to a file')
    logging.info(f"Hi, this is a test run.")

    population = MyPopulation(my_dataset)
    #population.create_random_straight(5)
    #population.create_random_forked(5)
    population.create_random_modules(1)

    iteration = population.iterate_epochs(epochs=1, training_epochs=5)

    print(iteration)


if __name__ == "__main__":
    
    test_run(global_configs, possible_components)