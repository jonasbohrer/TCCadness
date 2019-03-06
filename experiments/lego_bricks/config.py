class Config():
    """
    Set configurations here
    """

    import innvestigate.utils.visualizations as ivis
    from keras.preprocessing.image import ImageDataGenerator

    def __init__(self, modelfilename, method_n):
        self.methods = [ ("lrp.z",                           {},             ivis.heatmap,    "LRP-Z"),
                        ("lrp.epsilon",                     {"epsilon": 1}, ivis.heatmap,    "LRP-Epsilon"),
                        ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat"),
                        ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat"),
                        ("lrp.alpha_2_beta_1",              {},             ivis.heatmap,    "LRP-PresetAlpha2Beta1"),
                        ("lrp.alpha_2_beta_1_IB",           {},             ivis.heatmap,    "LRP-PresetAlpha2Beta1IB"),
                        ("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat"),
                        ("lrp.sequential_preset_b_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetBFlat")]
        self.method = self.methods[method_n]
        
        self.output_nodes = [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]
        self.analysis_mode = "all"
        self.models_dir = 'models/'+modelfilename+'/'
        self.figs_dir = self.models_dir+'figs/'+self.method[3]+'/'
        return None

    def get_lrp_image_set(self):

        """
        Set the images to be tested using LRP
        """
        test_datagen = ImageDataGenerator(rescale=1./255)

        validation_generator = test_datagen.flow_from_directory(
                'LEGO brick images/valid/',
                target_size=(200, 200),
                batch_size=32,
                class_mode='binary',
                seed= 1)

        images = [
            validation_generator[0][0][28:29],#0
            validation_generator[0][0][8:9],#1
            validation_generator[1][0][8:9],#2
            validation_generator[0][0][4:5],#3
            validation_generator[0][0][13:14],#4
            validation_generator[0][0][3:4],#5
            validation_generator[0][0][24:25],#6
            validation_generator[0][0][14:15],#7
            validation_generator[0][0][6:7],#8
            validation_generator[1][0][28:29],#9
            validation_generator[0][0][9:10],#10
            validation_generator[0][0][1:2],#11
            validation_generator[1][0][25:26],#12
            validation_generator[0][0][25:26],#13
            validation_generator[0][0][0:1],#14
            validation_generator[0][0][12:13],#15
        ]

        return images

class Model():
    """
    Define the model here
    """
    import keras
    import collections
    import os, copy, sys
    def create_model(self):
        input_shape = (200, 200, 3)
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
                    keras.layers.MaxPooling2D((2, 2)),
                    keras.layers.Conv2D(16, (3, 3), activation="relu"),
                    keras.layers.MaxPooling2D((2, 2)),
                    keras.layers.Flatten(),
                    keras.layers.Dense(128, activation="relu"),
                    keras.layers.Dense(16, activation="softmax"),
                    ])
        model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.001), metrics=["accuracy"])
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(10,
            kernel_size=(3, 3),
            activation='relu',
            kernel_initializer=keras.initializers.RandomUniform(),
            bias_initializer=keras.initializers.RandomUniform(),
            input_shape=(28,28,1)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(10,
            kernel_size=(3, 3),
            activation='relu',
            kernel_initializer=keras.initializers.RandomUniform(),
            bias_initializer=keras.initializers.RandomUniform()))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(100,
            kernel_initializer=keras.initializers.RandomUniform(),
            bias_initializer=keras.initializers.RandomUniform(),
            activation='relu'))
        model.add(keras.layers.Dense(10,
            kernel_initializer=keras.initializers.RandomUniform(),
            bias_initializer=keras.initializers.RandomUniform(),
            activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.SGD(momentum=0.1, lr=0.005),
            metrics=['accuracy'])"""

        return model

    def get_model_checkpoints(self):
        modelnames = []
        models = []

        for file_name in os.listdir(models_dir):
            if file_name.endswith('.h5'):
                modelnames.append(file_name)
        modelnames = sorted(modelnames, key=lambda x: (len(x), str.lower(x)))

        for modelname in modelnames:
            model = create_model()
            try:
                model.load_weights(models_dir+modelname)
            except: 
                model = keras.models.load_model(models_dir+modelname)
            models.append((modelname, model))

        models = collections.OrderedDict(models)
        return (models)