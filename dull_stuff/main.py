import keras, logging, random, pydot, copy, uuid
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum, auto
from typing import List
from keras.utils.vis_utils import plot_model
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import scale

input_configs = {
    "module_range" : ([1, 1], 'int'),
    "component_range" : ([1, 1], 'int')
}

global_configs = {
    "module_range" : ([3, 3], 'int'),
    "component_range" : ([5, 5], 'int')
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


class HistoricalMarker:
    
    def __init__(self):
        self.module_counter = 0
        self.blueprint_counter = 0
    
    def mark_module(self):
        self.module_counter += 1
        return self.module_counter

    def mark_blueprint(self):
        self.blueprint_counter += 1
        return self.blueprint_counter

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
    def __init__(self, representation, keras_component=None, complementary_component=None, keras_complementary_component=None, component_type=None):
        self.representation = representation
        self.keras_component = keras_component
        self.complementary_component = complementary_component
        self.keras_complementary_component = keras_complementary_component
        self.component_type = component_type
    
    def compare_to(self, component):
        if self.representation[0] == component.representation[0]:
            score = []
            for n in self.representation[1]:
                if self.representation[1][n] == component.representation[1][n]:
                    score.append(1)
                elif possible_components[self.component_type][1][n][1] == "int": 
                    ranges = possible_components[self.component_type][1][n][0]
                    scale = max(ranges) - min(ranges)
                    score.append(1 - abs(self.representation[1][n] - component.representation[1][n])/scale)
                else:
                    score.append(0)
            return sum(score)/len(score)
        else:
            return 0

    def get_layer_size(self):
        if self.component_type == "conv2d":
            return self.representation[1]["filters"]
        if self.component_type == "dense":
            return self.representation[1]["units"]

class Module(object):
    """
    Represents a set of one or more basic units of a topology.
    components: the most basic unit of a topology.
    """
    def __init__(self, components:dict, layer_type:ModuleComposition=ModuleComposition.INPUT, mark=None, component_graph=None):
        self.components = components
        self.component_graph = component_graph
        self.layer_type = layer_type
        self.mark = mark
        self.scores = [0,0]
        self.species = None
    
    def __getitem__(self, item):
        return self.components[item]
    
    def compare_to(self, module):
        nodes = self.component_graph.nodes()
        obj_module_node_defs = [module.component_graph.nodes[node]["node_def"] for node in module.component_graph.nodes()]
        
        if self.mark == module.mark:
            return 1
        else:
            score = []
            for node in nodes:
                scores = [nodes[node]["node_def"].compare_to(obj_node_def) for obj_node_def in obj_module_node_defs]
                score.append(max(scores))

            print(score)
            return sum(score)/len(score)

    def get_module_size(self):
        module_size = 0
        for node in self.component_graph.nodes():
            module_size += self.component_graph.nodes[node]["node_def"].get_layer_size()
        return module_size
    
    def get_kmeans_representation(self):
        node_count = len(self.component_graph.nodes())
        edge_count = len(self.component_graph.edges())
        module_size = self.get_module_size()
        scores = self.scores
        return node_count, edge_count, module_size, scores[0], scores[1]

    def update_scores(self, scores):
        self.scores = scores

class Blueprint:
    """
    Represents a topology made of modules.
    modules: modules composing a topology.
    """
    def __init__(self, modules:List[Module], input_shape=None, module_graph=None, mark=None):
        self.modules = modules
        self.input_shape = input_shape
        self.module_graph = module_graph
        self.mark = mark
        self.scores = [0,0]
        self.species = None

    def __getitem__(self, item):
        return self.modules[item]

    def get_blueprint_size(self):
        blueprint_size = 0
        for node in self.module_graph.nodes():
            blueprint_size += self.module_graph.nodes[node]["node_def"].get_module_size()
        return blueprint_size
    
    def get_kmeans_representation(self):
        node_count = len(self.module_graph.nodes())
        edge_count = len(self.module_graph.edges())
        blueprint_size = self.get_blueprint_size()
        scores = self.scores
        return node_count, edge_count, blueprint_size, scores[0], scores[1]

    def update_scores(self, scores):
        self.scores = scores
        for node in self.module_graph.nodes():
            self.module_graph.nodes[node]["node_def"].update_scores(scores)

class Species:
    """
    Represents a group of topologies with similarities.
    properties: a set of common properties defining a species.
    """
    def __init__(self, name=None, species_type=None, group=None, properties=None, starting_epoch=None):
        self.name = name
        self.species_type = species_type
        self.group = group
        self.properties = properties
        self.starting_epoch = starting_epoch

class Individual:
    """
    Represents a topology in the Genetic Algorithm scope.
    blueprint: representation of the topology.
    species: a grouping of topologies set according to similarity.
    birth: epoch the blueprint was created.
    parents: individuals used in crossover to generate this topology.
    """
       
    def __init__(self, blueprint:Blueprint, compiler=None, birth=None, parents=None, model=None, name=None):
        self.blueprint = blueprint
        self.compiler = compiler
        self.birth = birth
        self.parents = parents
        self.model = model
        self.name = name

    def generate(self, save_fig=True):
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

        if (save_fig):
            plt.tight_layout()
            plt.subplot(121)
            nx.draw(assembled_module_graph, nx.drawing.nx_agraph.graphviz_layout(assembled_module_graph, prog='dot'), with_labels=True, font_weight='bold', font_size=6)
            l,r = plt.xlim()
            plt.xlim(l-5,r+5)
            plt.savefig("./images/2_component_level_graph.png", format="PNG", bbox_inches="tight")
            plt.clf()

        logging.log(21, f"Generated the assembled graph: {assembled_module_graph.nodes()}")

        logging.info(f"Generating keras layers")

        #Adds Input layer
        logging.info(f"Adding the Input() layer")
        model_input = keras.layers.Input(shape=self.blueprint.input_shape)
        logging.log(21, f"Added {model_input}")

        #Garantees connections are defined in the correct order
        node_order = nx.algorithms.dag.topological_sort(assembled_module_graph)
        component_graph = assembled_module_graph

        # Iterate over the graph connecting keras layers
        for component_id in node_order:
            layer = []

            # Create a copy of the original layer so we don't have duplicate layers in the model in the future. Generates the keras layer now.
            component = copy.deepcopy(component_graph.nodes[component_id]["node_def"])
            component_def = component.representation[0]
            parameter_def = component.representation[1]
            component.keras_component = component_def(**parameter_def)
            component.keras_component.name = component.keras_component.name + "_" + uuid.uuid4().hex

            if component.complementary_component != None:
                component_def = component.complementary_component[0]
                parameter_def = component.complementary_component[1]
                component.keras_complementary_component = component_def(**parameter_def)
                component.keras_complementary_component.name = component.keras_complementary_component.name + "_" + uuid.uuid4().hex

            logging.log(21, f"{component_id}: Working on {component.keras_component.name}. Specs: {component.representation}")
            print(f"{component_id}: Working on {component.keras_component.name}. Specs: {component.representation}")

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
                # If dense connecting to previous conv, flatten
                if "conv2d" in predecessors[0].name and "dense" in component.keras_component.name:
                    logging.info(f"Adding a Flatten() layer between {predecessors[0].name} and {component.keras_component.name}")
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
                    logging.log(21, f"{component_id}: is {predecessors[predecessor][0].name} conv2d: {'conv2d' in predecessors[predecessor][0].name}. is {component.keras_component.name} dense: {'dense' in component.keras_component.name}")
                    if "conv2d" in predecessors[predecessor][0].name and "dense" in component.keras_component.name:
                        logging.info(f"Adding a Flatten() layer between {predecessors[predecessor][0].name} and {component.keras_component.name}")
                        layer = [keras.layers.Flatten()(predecessors[predecessor][-1])]
                        logging.log(21, f"{component_id}: Added {layer}")
                        predecessors[predecessor] = layer
                
                logging.info(f"Adding a Merge layer between {predecessors[0][0].name} and {predecessors[1][0].name}")
                merge_layer = keras.layers.concatenate([predecessors[0][-1], predecessors[1][-1]])
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
        self.model.compile(**self.compiler)
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
        
        #Update scores for blueprints (and underlying modules)
        self.blueprint.update_scores(scores)

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
    def __init__(self, datasets=None, individuals=[], blueprints=[], modules=[], hyperparameters=[], input_shape=None):
        self.datasets = datasets
        self.individuals = individuals
        self.blueprints = blueprints
        self.modules = modules
        self.historical_marker = HistoricalMarker()
        self.hyperparameters = hyperparameters
        self.module_species = None
        self.blueprint_species = None
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
            new_blueprint = GraphOperator().random_blueprint(global_configs, possible_components, possible_complementary_components, input_shape)

            #Create individual with the Blueprint
            new_individual = Individual(blueprint=new_blueprint, name=n, compiler=compiler)
            new_individuals.append(new_individual)

        self.individuals = new_individuals

    def create_module_population(self, size=1):
        """
        Creates a population of modules to be used in blueprint populations.
        Can be evolved over generations.
        """

        new_modules = []

        for n in range(size):
            mark = self.historical_marker.mark_module()
            new_module = GraphOperator().random_module(global_configs, 
                                                        possible_components, 
                                                        possible_complementary_components,
                                                        name=mark)
            new_module.mark = mark
            new_modules.append(new_module)

        #print(new_modules)
        self.modules = new_modules

    def create_blueprint_population(self, size=1):
        """
        Creates a population of blueprints to be used in individual populations.
        Can be evolved over generations.
        """

        new_blueprints = []

        for n in range(size):

            mark = self.historical_marker.mark_blueprint()

            #Create a blueprint
            input_shape = self.input_shape
            new_blueprint = GraphOperator().random_blueprint(global_configs,
                                                            possible_components, 
                                                            possible_complementary_components, 
                                                            input_shape,
                                                            node_content_generator=self.return_random_module,
                                                            args={},
                                                            name=mark)
            new_blueprint.mark = mark
            new_blueprints.append(new_blueprint)

        #print(new_blueprints)
        self.blueprints = new_blueprints
    
    def create_individual_population(self, size=1, compiler=None):
        """
        Creates a population of individuals to be compared.
        Can be evolved over generations.
        """

        new_individuals = []

        for n in range(size):

            #Create a blueprint
            input_shape = self.input_shape
            new_blueprint = self.return_random_blueprint()

            #Create individual with the Blueprint
            new_individual = Individual(blueprint=new_blueprint, name=n, compiler=compiler)
            new_individuals.append(new_individual)

        self.individuals = new_individuals

    def return_random_module(self):
        return random.choice(self.modules)

    def return_random_blueprint(self):
        return random.choice(self.blueprints)

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

    def apply_kmeans_speciation(self, items, n_clusters, species_type):
        """
        Apply KMeans to (re)start species
        """
        item_species = []
        representations = []

        for item in items:
            representations.append(item.get_kmeans_representation())
        
        classifier = KMeans(n_clusters=n_clusters, random_state=0)
        classifications = classifier.fit_predict(scale(representations))

        for species_name in range(len(classifier.cluster_centers_)):
            group = []
            for n in range(len(classifications)):
                if classifications[n] == species_name:
                    group.append(items[n])

            item_species.append(Species(name=species_name, species_type=species_type, group=group))

        for species in item_species:
            for item in species.group:
                item.species = species
        
        logging.log(21, f"KMeans generated {n_clusters} species using: {representations}.")
        
        (representations, classifications, item_species)
        return item_species, classifications

    def apply_centroid_speciation(self, items, species_list):
        """
        Apply the NearestCentroid method to assign members to existing species.
        Centroids are calculated based on existing species members and new members are assigned to the closest centroid.
        An accuracy threshold can be specified so new species are generated in case new members dont fit the existing centroid accordingly.
        """

        #Collect previous representations for centroid calculations.
        previous_member_representations = []
        previous_labels = []
        for species in species_list:
            previous_member_representations = previous_member_representations + [item.get_kmeans_representation() for item in species.group]
            previous_labels = previous_labels + [item.species.name for item in species.group]
        logging.log(21,(f"Previous species members: {previous_member_representations}. \nPrevious labels: {previous_labels}"))

        #Collect current representations for classification
        member_representations = []
        for item in items:
            member_representations.append(item.get_kmeans_representation())

        #Scale features using the whole data
        scaled_representations = scale(previous_member_representations + member_representations)
        #Select only speciated members to train the classifier
        scaled_previous_member_representations = scaled_representations[:len(previous_member_representations)]
        #Fit data to centroids
        classifier = NearestCentroid().fit(scaled_previous_member_representations, previous_labels)
        #Predict label to all data. New labels must be THE SAME as old labels, if they existed previously.
        all_classifications = classifier.predict(scaled_representations)
        logging.log(21,f"All Classifications: {all_classifications}")

        new_classifications = all_classifications[len(previous_member_representations):]
        logging.log(21,"Old Classifications: {previous_labels}. \nNew Classifications: {new_classifications}")

        #Update species members
        for species in species_list:
            group = []
            for n in range(len(new_classifications)):
                if new_classifications[n] == species.name:
                    group.append(items[n])

            species.group = group

        #Update species info in items
        for species in species_list:
            for item in species.group:
                item.species = species

        return None

    def create_module_species(self, n_clusters):
        """
        Divides the modules in groups according to similarity.
        """

        module_species, module_classifications = self.apply_kmeans_speciation(self.modules, n_clusters, species_type="module")

        self.module_species = module_species

        logging.log(21, f"Created {n_clusters} module species.")
        for species in module_species:
            logging.log(21, f"Species {species.name}: {species.group}")

        return (module_classifications)

    def create_blueprint_species(self, n_clusters):
        """
        Divides the blueprints in groups according to similarity.
        """

        blueprint_species, blueprint_classifications = self.apply_kmeans_speciation(self.blueprints, n_clusters, species_type="blueprints")
        
        self.blueprint_species = blueprint_species

        logging.log(21, f"Created {n_clusters} blueprint species.")
        for species in blueprint_species:
            logging.log(21, f"Species {species.name}: {species.group}")

        return (blueprint_classifications)
    
    def update_module_species(self):
        """
        Divides the modules in groups according to similarity.
        """

        self.apply_centroid_speciation(self.modules, self.module_species)

        return None

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

class GraphOperator:

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
        nx.draw(graph, nx.drawing.nx_agraph.graphviz_layout(graph, prog='dot'), with_labels=True, font_weight='bold', font_size=6)
        plt.tight_layout()
        plt.savefig(f"./images/{filename}", format="PNG", bbox_inches="tight")
        plt.clf()

    def random_component(self, possible_components, possible_complementary_components = None):
        component_type = random.choice(list(possible_components))
        component_def, possible_parameters = possible_components[component_type]

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

        new_component = Component(representation=[component_def, parameter_def],
                                    keras_component=component_def(**parameter_def),
                                    complementary_component=complementary_component,
                                    keras_complementary_component=keras_complementary_component,
                                    component_type=component_type)
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
                leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                root_node = min([node for node in new_graph.nodes() if new_graph.in_degree(node) == 0])
                leaf_nodes.remove(node)

                while (len(leaf_nodes) > 0):
                    if len(leaf_nodes) <= 2:
                        leaf_node = random.choice(leaf_nodes)
                        new_graph.add_edge(leaf_node, node)
                    else:
                        leaf_nodes.append(root_node)
                        random_node1 = random.choice(leaf_nodes)
                        simple_paths = [node for path in nx.all_simple_paths(new_graph, root_node, random_node1) for node in path]
                        leaf_nodes.remove(random_node1)
                        random_node2 = random.choice(leaf_nodes)
                        if (new_graph.in_degree(random_node2) >= 1 and random_node2 not in simple_paths and random_node2 != root_node):
                            new_graph.add_edge(random_node1, random_node2)
                    leaf_nodes = [leaf_node for leaf_node in new_graph.nodes() if new_graph.out_degree(leaf_node)==0]
                    leaf_nodes.remove(node)

        return new_graph

    def random_module(self, global_configs, possible_nodes, possible_complementary_components, name=0):

        node_range = self.random_parameter_def(global_configs, "component_range")
        logging.log(21, f"Generating {node_range} components")
        print(f"Generating {node_range} components")

        graph = self.random_graph(node_range=node_range,
                                            node_content_generator=self.random_component,
                                            args = {"possible_components": possible_nodes,
                                                    "possible_complementary_components": possible_complementary_components})

        self.save_graph_plot(f"module_{name}_{self.count}_module_internal_graph.png", graph)
        self.count+=1
        new_module = Module(None, ModuleComposition.INTERMED, component_graph=graph)

        return new_module

    def random_blueprint(self, global_configs, possible_components, possible_complementary_components, input_shape, node_content_generator=None, args={}, name=0):

        node_range = self.random_parameter_def(global_configs, "module_range")
        logging.log(21, f"Generating {node_range} modules")
        print(f"Generating {node_range} modules")

        if (node_content_generator == None):
            node_content_generator = self.random_module
            args = {"global_configs": global_configs,
                                                    "possible_nodes": possible_components,
                                                    "possible_complementary_components": possible_complementary_components}


        input_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": input_configs,
                                                    "possible_nodes": possible_inputs,
                                                    "possible_complementary_components": None})
        self.save_graph_plot(f"blueprint_{name}_input_module.png", input_node)

        intermed_graph = self.random_graph(node_range=node_range,
                                            node_content_generator=node_content_generator,
                                            args = args)
        self.save_graph_plot(f"blueprint_{name}_intermed_module.png", intermed_graph)

        output_node = self.random_graph(node_range=1,
                                            node_content_generator=self.random_module,
                                            args = {"global_configs": output_configs,
                                                    "possible_nodes": possible_outputs,
                                                    "possible_complementary_components": None})
        self.save_graph_plot(f"blueprint_{name}_output_module.png", output_node)

        graph = nx.union(input_node, intermed_graph, rename=("input-", "intermed-"))
        graph = nx.union(graph, output_node, rename=(None, "output-"))
        graph.add_edge("input-0", "intermed-0")
        graph.add_edge(f"intermed-{max(intermed_graph.nodes())}", "output-0")
        self.save_graph_plot(f"blueprint_{name}_module_level_graph.png", graph)

        new_blueprint = Blueprint(None, input_shape, module_graph=graph)

        return new_blueprint

    def mutate_by_node_removal(self, graph):

        new_graph = graph.copy()

        candidate_nodes = [node for node in new_graph.nodes() if new_graph.out_degree(node) > 0 and new_graph.in_degree(node) > 0]
        selected_node = random.choice(candidate_nodes)
        
        # Removing a node
        if len(list(new_graph.predecessors(selected_node))) > 0 and len(list(new_graph.successors(selected_node))) > 0:
            predecessors = new_graph.predecessors(selected_node)
            successors = new_graph.successors(selected_node)

            new_edges = [(p,s) for p in predecessors for s in successors]

            new_graph.remove_node(selected_node)
            new_graph.add_edges_from(new_edges)

        return new_graph
    
    def mutate_by_node_addition_in_edges(self, graph, args=None):

        new_graph = graph.copy()

        # "working around" the bad naming decisions I make in life.
        try:
            node = int(max(new_graph.nodes())) + 1
        except:
            node = "intermed-" + str(max([int(node.split('-')[1]) for node in new_graph.nodes() if 'input' not in node and 'output' not in node]) + 1)

        candidate_edges = [edge for edge in new_graph.edges()]
        selected_edge = random.choice(candidate_edges)
        
        # Adding a node 
        predecessor = selected_edge[0]
        successor = selected_edge[1]

        node_def = self.random_component(**args)
        new_graph.add_node(node, node_def=node_def)
        new_graph.remove_edge(predecessor, successor)

        new_graph.add_edge(predecessor, node)
        new_graph.add_edge(node, successor)

        return new_graph
    
    def mutate_by_node_addition_outside_edges(self, graph, args=None):

        new_graph = graph.copy()

        try:
            node = int(max(new_graph.nodes())) + 1
        except:
            node = "intermed-" + str(max([int(node.split('-')[1]) for node in new_graph.nodes() if 'input' not in node and 'output' not in node]) + 1)

        node_def = self.random_component(**args)

        # Select nodes that are not outputs
        candidate_predecessor_nodes = [node for node in new_graph.nodes() if new_graph.out_degree(node) > 0]

        #Select random predecessor
        predecessor = random.choice(candidate_predecessor_nodes)
        starting_node = min([node for node in new_graph.nodes() if new_graph.in_degree(node) == 0])
        simple_paths = [node for path in nx.all_simple_paths(new_graph, starting_node, predecessor) for node in path]

        # Select nodes that are not inputs and have at most 1 inputs (merge only supports 2 input layers)
        candidate_successor_nodes = [node for node in new_graph.nodes() if new_graph.in_degree(node) == 1 and node not in simple_paths]

        # If no successors available just create the node between an existing edge.
        if candidate_successor_nodes == []:
            successor = random.choice(new_graph.successors(predecessor))
            new_graph.remove_edge(predecessor, successor)
        else:
            successor = random.choice(candidate_successor_nodes)

        new_graph.add_node(node, node_def=node_def)
        new_graph.add_edge(predecessor, node)
        new_graph.add_edge(node, successor)

        return new_graph

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
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], input_shape)
                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint, compiler=new_compiler)
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
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], input_shape)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint, compiler=new_compiler)
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
                new_blueprint = Blueprint([new_input_module, new_intermed_module, new_output_module], input_shape)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint, compiler=new_compiler)
                new_individuals.append(new_individual)

            self.individuals = new_individuals

        def create_fixed_blueprints(self, size=1):

            new_individuals = []

            for n in range(size):
                #Create input module
                new_input_module = GraphOperator().random_module(input_configs, possible_inputs)
              
                #Create intermediate module
                new_intermed_module1 = GraphOperator().random_module(global_configs, possible_components)
                new_intermed_module2 = GraphOperator().random_module(global_configs, possible_components)

                #Create output module
                new_output_module = GraphOperator().random_module(output_configs, possible_outputs)

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

                new_blueprint = Blueprint([new_input_module, new_intermed_module1, new_intermed_module2, new_output_module], input_shape, module_graph=module_graph)

                #Create individual with the Blueprint
                new_individual = Individual(blueprint=new_blueprint, compiler=new_compiler)
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

def run_cifar10_nopop(global_configs, possible_components, population_size, epochs, training_epochs):
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
    population.create_random_blueprints(population_size, compiler=compiler)

    iteration = population.iterate_epochs(epochs=epochs, training_epochs=training_epochs, validation_split=0.15)

    print("Best fitting: ", iteration)

def run_cifar10_tests(global_configs, possible_components, population_size, epochs, training_epochs):
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

    n_blueprint_species = 3
    n_module_species = 3

    ###########
    # MODULES #
    ###########
    
    # Start with random modules
    population.create_module_population(10)

    #Test: original module selection
    module = random.choice(population.modules)
    GraphOperator().save_graph_plot("test_module_original.png", module.component_graph)
    args = {"possible_components": possible_components, "possible_complementary_components": possible_complementary_components}

    #Test: mutation by removal
    print(f"module: {module.mark}, graph nodes: {module.component_graph.nodes()}, \ngraph edges: {module.component_graph.edges()}")
    mutated_graph = GraphOperator().mutate_by_node_removal(module.component_graph)
    print(f"mutation by removal; edges: {mutated_graph.edges()}")
    GraphOperator().save_graph_plot("test_module_mutate_by_node_removal.png", mutated_graph)

    #Test: mutation by addition
    print(f"module: {module.mark}, graph nodes: {module.component_graph.nodes()}, \ngraph edges: {module.component_graph.edges()}")
    mutated_graph = (GraphOperator().mutate_by_node_addition_in_edges(module.component_graph, args=args))
    print(f"mutation by addition; edges: {mutated_graph.edges()}")
    GraphOperator().save_graph_plot("test_module_mutate_by_node_addition_in_edges.png", mutated_graph)

    #Test: mutation by addition 2
    print(f"module: {module.mark}, graph nodes: {module.component_graph.nodes()}, \ngraph edges: {module.component_graph.edges()}")
    mutated_graph = (GraphOperator().mutate_by_node_addition_outside_edges(module.component_graph, args = args))
    print(f"mutation by addition 2; edges: {mutated_graph.edges()}")
    GraphOperator().save_graph_plot("test_module_mutate_by_node_addition_outside_edges.png", mutated_graph)

    #Start module species
    population.create_module_species(n_module_species)
    for species in population.module_species:
        print(f"Initial Module Species {species.name}: {[item.mark for item in species.group]}")
    print(f"Modules: {[item.mark for item in population.modules]}. \nSpecies: {[item.species.name for item in population.modules]}")

    #Add unspeciated members
    current_population = population.modules
    population.create_module_population(5)
    population.modules = population.modules + current_population

    #Speciate
    population.update_module_species()
    for species in population.module_species:
        print(f"Re-Speciated Module Species {species.name}: {[item.mark for item in species.group]}")
    print(f"Modules: {[item.mark for item in population.modules]}. \nSpecies: {[item.species.name for item in population.modules]}")

    exit(0)


    ##############
    # BLUEPRINTS #
    ##############

    # Start with random blueprints from modules
    population.create_blueprint_population(5)

    #Test: original blueprint selection
    blueprint = random.choice(population.blueprints)
    GraphOperator().save_graph_plot("test_blueprint_original.png", blueprint.module_graph)
    args = {"possible_components": possible_components, "possible_complementary_components": possible_complementary_components}

    #Test: mutation by removal
    print(f"blueprint: {blueprint.mark}, graph nodes: {blueprint.module_graph.nodes()}, \ngraph edges: {blueprint.module_graph.edges()}")
    mutated_graph = GraphOperator().mutate_by_node_removal(blueprint.module_graph)
    print(f"mutation by removal; edges: {mutated_graph.edges()}")
    GraphOperator().save_graph_plot("test_blueprint_mutate_by_node_removal.png", mutated_graph)

    #Test: mutation by addition
    print(f"blueprint: {blueprint.mark}, graph nodes: {blueprint.module_graph.nodes()}, \ngraph edges: {blueprint.module_graph.edges()}")
    mutated_graph = (GraphOperator().mutate_by_node_addition_in_edges(blueprint.module_graph, args=args))
    print(f"mutation by addition; edges: {mutated_graph.edges()}")
    GraphOperator().save_graph_plot("test_blueprint_mutate_by_node_addition_in_edges.png", mutated_graph)

    #Test: mutation by addition 2
    print(f"blueprint: {blueprint.mark}, graph nodes: {blueprint.module_graph.nodes()}, \ngraph edges: {blueprint.module_graph.edges()}")
    mutated_graph = (GraphOperator().mutate_by_node_addition_outside_edges(blueprint.module_graph, args = args))
    print(f"mutation by addition 2; edges: {mutated_graph.edges()}")
    GraphOperator().save_graph_plot("test_blueprint_mutate_by_node_addition_outside_edges.png", mutated_graph)

    #Start blueprint species
    population.create_blueprint_species(n_blueprint_species)
    for species in population.blueprint_species:
        print(f"Initial Blueprint Species {species.name}: {[item.mark for item in species.group]}")
    print(f"Blueprints: {[item.mark for item in population.blueprints]}. \nSpecies: {[item.species.name for item in population.blueprints]}")

    #Add unspeciated members
    current_population = population.blueprints
    population.create_blueprint_population(5)
    population.blueprints = population.blueprints + current_population

    #Speciate
    population.update_blueprint_species()
    for species in population.blueprint_species:
        print(f"Re-Speciated Blueprint Species {species.name}: {[item.mark for item in species.group]}")
    print(f"Blueprints: {[item.mark for item in population.blueprints]}. \nSpecies: {[item.species.name for item in population.blueprints]}")

    ###############
    # POPULATIONS #
    ###############

    # Start with random blueprints from modules
    population.create_individual_population(5, compiler)

    iteration = population.iterate_epochs(epochs=epochs, training_epochs=training_epochs, validation_split=0.15)
    
    exit(0)

    print("Best fitting: ", iteration)
  
if __name__ == "__main__":
    
    #test_run(global_configs, possible_components)
    #run_cifar10_nopop(global_configs, possible_components, population_size=5, epochs=5, training_epochs=5)
    run_cifar10_tests(global_configs, possible_components, population_size=5, epochs=1, training_epochs=1)