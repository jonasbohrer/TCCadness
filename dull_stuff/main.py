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
    "module_range" : ([1, 3], 'int'),
    "component_range" : ([1, 4], 'int')
}

output_configs = {
    "module_range" : ([1, 1], 'int'),
    "component_range" : ([1, 1], 'int')
}

possible_inputs = {
    "conv2d": (keras.layers.Conv2D, {"filters": ([48,48], 'int'), "kernel_size": ([3], 'list'), "activation": (["relu"], 'list')})
}

possible_components = {
    "conv2d": (keras.layers.Conv2D, {"filters": ([8, 48], 'int'), "kernel_size": ([1], 'list'), "strides": ([1], 'list'), "data_format": (['channels_last'], 'list'), "padding": (['same'], 'list')}),
    #"dense": (keras.layers.Dense, {"units": ([8, 48], 'int')})
}

possible_outputs = {
    "dense": (keras.layers.Dense, {"units": ([10,10], 'int'), "activation": (["softmax"], 'list')})
}

possible_complementary_components = {
    #"maxpooling2d": (keras.layers.MaxPooling2D, {"pool_size": ([2, 3], 'list')}),
    "dropout": (keras.layers.Dropout, {"rate": ([0, 0.7], 'float')})
}

class HistoricalMarking:
    def __init__(self):
        counter = 0
    
    def generate(self):
        counter += 1
        return counter

class NameGenerator:
    def __init__(self):
        self.counter = 0
    
    def generate(self):
        self.counter += 1
        return self.counter

class ModuleComposition(Enum):
    INPUT = "input"
    INTERMED = "intermed"
    CONV = "conv2d"
    DENSE = "dense"
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
    def __init__(self, representation, keras_component=None, complementary_component=None, keras_complementary_component=None):
        self.representation = representation
        self.keras_component = keras_component
        self.historical_marking = 0
        self.complementary_component = complementary_component
        self.keras_complementary_component = keras_complementary_component

class Module(object):
    """
    Represents a set of one or more basic units of a topology.
    components: the most basic unit of a topology.
    """
    def __init__(self, components:dict, layer_type:ModuleComposition=ModuleComposition.INPUT, component_graph=None):
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
    def __init__(self, modules:List[Module], compiler=None, input_shape=None, module_graph=None):
        self.modules = modules
        self.compiler = compiler
        self.input_shape = input_shape
        self.module_graph = module_graph

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
       
    def __init__(self, blueprint:Blueprint, birth=None, parents=None, model=None, name=None):
        self.blueprint = blueprint
        self.birth = birth
        self.parents = parents
        self.model = model
        self.name = name

    def generate(self):
        """
        Returns the keras model representing of the topology.
        """

        layer_map = {}
        module_graph = self.blueprint.module_graph

        assembled_module_graph = nx.DiGraph()
        output_nodes = {}

        for node in module_graph.nodes():
            assembled_module_graph = nx.union(assembled_module_graph, module_graph.nodes[node]["node_def"].component_graph, rename=(None, f'{node}-'))
            output_nodes[node] = (max(module_graph.nodes[node]["node_def"].component_graph.nodes()))

        for node in module_graph.nodes():
            for successor in module_graph.successors(node):
                assembled_module_graph.add_edge(f'{node}-{output_nodes[node]}', f'{successor}-0')

        plt.subplot(121)
        nx.draw(assembled_module_graph, nx.drawing.nx_agraph.graphviz_layout(assembled_module_graph, prog='dot'), with_labels=True, font_weight='bold')
        plt.get_current_fig_manager().full_screen_toggle()
        plt.tight_layout()
        plt.savefig("./images/2_component_level_graph.png", format="PNG")
        plt.clf()

        logging.log(21, f"Generated the assembled graph: {assembled_module_graph.nodes()}")

        logging.info(f"Generating keras layers")
        logging.info(f"Adding the Input() layer")
        model_input = keras.layers.Input(shape=self.blueprint.input_shape)
        logging.log(21, f"Added {model_input}")

        component_graph = assembled_module_graph

        # Iterate over the graph including nodes
        for component_id in component_graph.nodes():
            layer = []
            component = component_graph.nodes[component_id]["node_def"]

            # If the node has no inputs, then use the Model Input as layer input
            if component_graph.in_degree(component_id) == 0:
                layer.append(component.keras_component(model_input))
                logging.log(21, f"{component_id}: Added {layer}")
                if component.complementary_component != None:
                    layer.append(component.keras_complementary_component(layer[0]))
                    logging.log(21, f"{component_id}: Added complement {layer}")
            
            # Else, if only one input, include it as layer input
            elif component_graph.in_degree(component_id) == 1:
                predecessors = [layer_map[predecessor_id] for predecessor_id in component_graph.predecessors(component_id)][0]
                logging.log(21, f"{component_id}: is {predecessors[0].name} conv2d: {'conv2d' in predecessors[0].name}. is {component.keras_component.name} dense: {'dense' in component.keras_component.name}")
                if "conv2d" in predecessors[0].name and "dense" in component.keras_component.name:
                    logging.info(f"Adding a Flatten() layer")
                    layer = [keras.layers.Flatten()(predecessors[-1])]
                    logging.log(21, f"{component_id}: Added {layer}")
                    predecessors = layer
                layer = [component.keras_component(predecessors[-1])]
                logging.log(21, f"{component_id}: Added {layer}")
                if component.complementary_component != None:
                    layer.append(component.keras_complementary_component(layer[0]))
                    logging.log(21, f"{component_id}: Added complement {layer}")
            
            # Else, if two inputs, merge them and use the merge as layer input
            elif component_graph.in_degree(component_id) == 2:
                predecessors = [layer_map[predecessor_id] for predecessor_id in component_graph.predecessors(component_id)]
                for predecessor in range(len(predecessors[0])):
                    if "conv2d" in predecessors[predecessor][0].name and "dense" in component.keras_component.name:
                        logging.info(f"Adding a Flatten() layer")
                        layer = [keras.layers.Flatten()(predecessors[predecessor][-1])]
                        logging.log(21, f"{component_id}: Added {layer}")
                        predecessors[predecessor] = layer
                
                logging.info(f"Adding a Merge layer")
                merge_layer = keras.layers.concatenate([predecessors[0][0], predecessors[1][0]])
                logging.log(21, f"{component_id}: Added {merge_layer}")
                layer = [component.keras_component(merge_layer)]
                logging.log(21, f"{component_id}: Added {layer}")
                if component.complementary_component != None:
                    layer.append(component.keras_complementary_component(layer[-1]))
                    logging.log(21, f"{component_id}: Added complement {layer}")

            # Store model layers as a reference while the model is still not assembled 
            layer_map[component_id] = layer

        # Assemble model
        logging.log(21, layer_map)
        self.model = keras.models.Model(inputs=model_input, outputs=layer_map[max(layer_map)])
        self.model.compile(**self.blueprint.compiler)
        plot_model(self.model, to_file='./images/3_layer_level_graph.png', show_shapes=True, show_layer_names=True)

    def fit(self, input_x, input_y, epochs=1, validation_split=0.15):
        """
        Fits the keras model representing the topology.
        """

        logging.info(f"Fitting one individual for {epochs} epochs")
        self.generate()
        fitness = self.model.fit(input_x, input_y, epochs=epochs, validation_split=validation_split, batch_size=512)
        print(fitness)

        return fitness

    def score(self, test_x, test_y):
        """
        Scores the keras model representing the topology.

        returns test_loss, test_acc
        """
        
        logging.info(f"Scoring one individual")
        scores = self.model.evaluate(test_x, test_y, verbose=1)

        return scores

    def extract_blueprint(self):
        """
        Generates a Blueprint description of the self.model variable.
        """
        pass

class Population:
    """
    Represents the population containing multiple individual topologies and their correlations.
    """
    def __init__(self, datasets=None, individuals=[], modules=[], hyperparameters=[], species=[], groups=[], input_shape=None):
        self.datasets = datasets
        self.individuals = individuals
        self.modules = modules
        self.hyperparameters = hyperparameters
        self.species = species
        self.groups = groups
        self.input_shape = input_shape

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

    def create_random_blueprints(self, size=1, compiler=None):

        new_individuals = []

        for n in range(size):

            #Create a blueprint
            input_shape = self.input_shape
            new_compiler = compiler
            new_blueprint = Generator().random_blueprint(global_configs, possible_components, possible_complementary_components, input_shape, new_compiler)

            #Create individual with the Blueprint
            new_individual = Individual(blueprint=new_blueprint, name=n)
            new_individuals.append(new_individual)

        self.individuals = new_individuals

    def return_individual(self, name):
        for individual in self.individuals:
            if individual.name == name:
                return individual

        return False

    def iterate_fitness(self, training_epochs=1, validation_split=0.15):
        """
        Fits the individuals and generates scores.

        returns a list composed of [invidual name, test scores, training history]
        """

        logging.info(f"Iterating fitness over {len(self.individuals)} individuals")
        iteration = []

        #(batch, channels, rows, cols)
        input_x = self.datasets.training[0]
        input_y = self.datasets.training[1]
        test_x = self.datasets.test[0]
        test_y = self.datasets.test[1]

        for individual in self.individuals:
            history = individual.fit(input_x, input_y, training_epochs, validation_split)
            score = individual.score(test_x, test_y)
            iteration.append([individual.name, score, history])

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

    def iterate_epochs(self, epochs=1, training_epochs=1, validation_split=0.15):
        """
        Manages epoch iterations, applying the genetic algorithm in fact.

        returns a list of epoch iterations
        """

        logging.info(f"Iterating over {epochs} epochs")
        iterations = []
        best_scores = []

        for epoch in range(epochs):
            logging.info(f" -- Iterating epoch {epoch} -- ")
            print(f" -- Iterating epoch {epoch} -- ")

            iteration = self.iterate_fitness(training_epochs, validation_split)
            iterations.append(iteration)
            # [name, score[test_loss, test_val], history]

            best_fitting = max(iteration, key=lambda x: (x[1][1], x[1][0]))
            print(best_fitting)
            
            self.return_individual(best_fitting[0]).model.save(f"./models/best_epoch_{epoch}.h5")

            best_scores.append(best_fitting)

        return best_scores

class Generator():

    count=0

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

    @staticmethod
    def save_graph_plot(filename, graph):
        plt.subplot(121)
        nx.draw(graph, nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot'), with_labels=True, font_weight='bold')
        plt.tight_layout()
        plt.savefig(f"./images/{filename}", format="PNG", figsize=(8.0, 5.0))
        plt.clf()

    def random_component(self, possible_components, possible_complementary_components = None):
        component_def, possible_parameters = possible_components[random.choice(list(possible_components))]

        parameter_def = {}
        for parameter_name in possible_parameters:
            parameter_def[parameter_name] = self.random_parameter_def(possible_parameters, parameter_name)

        if possible_complementary_components != None:
            compl_component_def, possible_compl_parameters = possible_complementary_components[random.choice(list(possible_complementary_components))]

            compl_parameter_def = {}
            for parameter_name in possible_compl_parameters:
                compl_parameter_def[parameter_name] = self.random_parameter_def(possible_compl_parameters, parameter_name)
            complementary_component = [compl_component_def, compl_parameter_def]
            keras_complementary_component = compl_component_def(**compl_parameter_def)
        else:
            complementary_component = None
            keras_complementary_component = None

        new_component = Component([component_def, parameter_def], component_def(**parameter_def), complementary_component=complementary_component, keras_complementary_component=keras_complementary_component)
        return new_component

    def random_graph(self, node_range, node_content_generator, args=None):

        new_graph = nx.DiGraph()

        for node in range(node_range):
            node_def = node_content_generator(**args)
            new_graph.add_node(node, node_def=node_def)

            if node == 0:
                pass
            elif node > 0 and (node < node_range-1 or node_range <= 2):
                precedent = random.randint(0, node-1)
                new_graph.add_edge(precedent, node)
            elif node == node_range-1:
                leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0 or leaf_node==0]
                leaf_nodes.remove(node)

                while (len(leaf_nodes) > 0):
                    if len(leaf_nodes) <= 2:
                        leaf_node = random.choice(leaf_nodes)
                        new_graph.add_edge(leaf_node, node)
                    else:
                        leaf_nodes.append(0)
                        random_node1 = random.choice(leaf_nodes)
                        leaf_nodes.remove(random_node1)
                        random_node2 = random.choice(leaf_nodes)
                        if (new_graph.in_degree(random_node2)>1 and random_node2 not in new_graph.successors(random_node1) and random_node2 != 0):
                            new_graph.add_edge(random_node1, random_node2)
                    leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                    leaf_nodes.remove(node)

        return new_graph

    def random_module(self, global_configs, possible_nodes, possible_complementary_components):

        node_range = self.random_parameter_def(global_configs, "component_range")
        logging.log(21, f"Generating {node_range} components")
        print(f"Generating {node_range} components")

        graph = self.random_graph(node_range=node_range,
                                            node_content_generator=self.random_component,
                                            args = {"possible_components": possible_nodes,
                                                    "possible_complementary_components": possible_complementary_components})

        self.save_graph_plot(f"0_{self.count}_module_internal_graph.png", graph)
        self.count+=1
        new_module = Module(None, ModuleComposition.INTERMED, component_graph=graph)

        return new_module
    
    def random_blueprint(self, global_configs, possible_components, possible_complementary_components, input_shape, compiler):

        node_range = self.random_parameter_def(global_configs, "module_range")
        logging.log(21, f"Generating {node_range} modules")
        print(f"Generating {node_range} modules")

        input_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": input_configs,
                                                    "possible_nodes": possible_inputs,
                                                    "possible_complementary_components": None})
        self.save_graph_plot("1_1_input_module.png", input_node)

        intermed_graph = self.random_graph(node_range=node_range,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": global_configs,
                                                    "possible_nodes": possible_components,
                                                    "possible_complementary_components": possible_complementary_components})
        self.save_graph_plot("1_2_intermed_module.png", intermed_graph)

        output_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": output_configs,
                                                    "possible_nodes": possible_outputs,
                                                    "possible_complementary_components": None})
        self.save_graph_plot("1_3_output_module.png", output_node)

        graph = nx.union(input_node, intermed_graph, rename=("input-", "intermed-"))
        graph = nx.union(graph, output_node, rename=(None, "output-"))
        graph.add_edge("input-0", "intermed-0")
        graph.add_edge(f"intermed-{max(intermed_graph.nodes())}", "output-0")
        self.save_graph_plot("1_module_level_graph.png", graph)

        new_blueprint = Blueprint(None, compiler, input_shape, module_graph=graph)

        return new_blueprint

def test_run(global_configs, possible_components):
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
                new_input_module = Module([[None, None, new_input_conv_component]], ModuleComposition.INPUT)
                new_intermed_module = Module([[None, None, new_conv_component]], ModuleComposition.INTERMED)
                new_output_module = Module([[None, None, new_softmax_component]], ModuleComposition.OUTPUT)
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
                new_input_module = Module([[None, None, new_input_conv_component]], ModuleComposition.INPUT)
                
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
                                                    [2, None, self.random_component()]], ModuleComposition.INPUT)
                    return new_module

                #new_input_module = new_module()
                #Create intermediate module
                new_intermed_module = Module([[None, 1, self.random_component()],
                                                [0, 2, self.random_component()],
                                                [1, 3, self.random_component()],
                                                [2, None, self.random_component()]], ModuleComposition.INTERMED)

                #Create output module
                new_softmax_component = Component(None, keras.layers.Dense(2, activation="softmax"),)
                new_output_module = Module([[None, None, new_softmax_component]], ModuleComposition.OUTPUT)

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
                new_input_module = Module([[None, None, new_input_conv_component]], ModuleComposition.INPUT)
              
                #Create intermediate module
                new_intermed_module = Module([[None, 1, self.random_component()],
                                                [0, 3, self.random_component()],
                                                [0, 3, self.random_component()],
                                                [(1,2), None, self.random_component()]], ModuleComposition.INTERMED)

                #Create output module
                new_softmax_component = Component(None, keras.layers.Dense(2, activation="softmax"),)
                new_output_module = Module([[None, None, new_softmax_component]], ModuleComposition.OUTPUT)

                #Set the compiler
                new_compiler = {"loss":"sparse_categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}
                input_shape = (8, 8, 1)
                #Create the blueprint combining modules
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], new_compiler, input_shape)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

        def create_fixed_blueprints(self, size=1):

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
                #Create the blueprint combining modules
                module_graph = nx.DiGraph()
                
                module_graph.add_node(0, module_def=new_input_module)
                module_graph.add_node(1, module_def=new_intermed_module1)
                module_graph.add_node(2, module_def=new_intermed_module2)
                module_graph.add_node(3, module_def=new_output_module)
                module_graph.add_edge(0, 1)
                module_graph.add_edge(0, 2)
                module_graph.add_edge(1, 3)
                module_graph.add_edge(2, 3)

                new_blueprint = Blueprint([new_input_module, new_intermed_module1, new_intermed_module2, new_output_module], new_compiler, input_shape, module_graph=module_graph)

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

    population = MyPopulation(my_dataset, input_shape = (8, 8, 1))
    population.create_random_blueprints(1)

    iteration = population.iterate_epochs(epochs=1, training_epochs=5)

    print(iteration)

def run_cifar10(global_configs, possible_components, population_size, epochs, training_epochs):
    from keras.datasets import cifar10

    possible_inputs = {
        "conv2d": (keras.layers.Conv2D, {"filters": ([48,48], 'int'), "kernel_size": ([3], 'list'), "activation": (["relu"], 'list')})
    }

    possible_components = {
        "conv2d": (keras.layers.Conv2D, {"filters": ([8, 48], 'int'), "kernel_size": ([1], 'list'), "strides": ([1], 'list'), "data_format": (['channels_last'], 'list'), "padding": (['same'], 'list')}),
        #"dense": (keras.layers.Dense, {"units": ([8, 48], 'int')})
    }

    possible_outputs = {
        "dense": (keras.layers.Dense, {"units": ([1,1], 'int'), "activation": (["softmax"], 'list')})
    }
    
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = (x_train[0:1000], y_train[0:1000]), (x_test[0:100], y_test[0:100])
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    batch_size = 32
    num_classes = 10
    data_augmentation = False
    num_predictions = 20

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    my_dataset = Datasets(training=[x_train, y_train], test=[x_test, y_test])

    logging.basicConfig(filename='test.log',
                        filemode='w+', level=logging.DEBUG,
                        format='%(levelname)s - %(asctime)s: %(message)s')
    logging.addLevelName(21, "TOPOLOGY")

    logging.warning('This will get logged to a file')
    logging.info(f"Hi, this is a test run.")

    compiler = {"loss":"categorical_crossentropy", "optimizer":keras.optimizers.Adam(lr=0.005), "metrics":["accuracy"]}

    population = Population(my_dataset, input_shape=x_train.shape[1:])
    population.create_random_blueprints(population_size, compiler)

    iteration = population.iterate_epochs(epochs=epochs, training_epochs=training_epochs, validation_split=0.15)

    print("Best fitting: ", iteration)

    
if __name__ == "__main__":
    
    #test_run(global_configs, possible_components)
    run_cifar10(global_configs, possible_components, population_size=5, epochs=5, training_epochs=5)