import keras, os, sys
import matplotlib.pyplot as plot
from keras.preprocessing.image import ImageDataGenerator

input_shape = (200, 200, 3)

model = keras.models.Sequential([
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
model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(lr=0.005), metrics=["accuracy"])

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
        'LEGO brick images/train/',
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary',
        seed=1)

#train_generator[batch_n = total/batch_size][0=image][x][y][channel]
#train_generator[batch_n = total/batch_size][1=class][indv; size = batch_size]

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        'LEGO brick images/valid/',
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary')

#train_generator[batch_n = total/batch_size][0=image][x][y][channel]
#train_generator[batch_n = total/batch_size][1=class][indv; size = batch_size]
#print(len(validation_generator[0][0][0]))

"""model.fit_generator(
        train_generator,
        steps_per_epoch=200,
        epochs=1,
        validation_data=validation_generator,
        validation_steps=5)"""

modelfilename = sys.argv[1] if sys.argv[1] != None else "test_model"
checkpoints = 1
epochs = 50
steps_per_epoch = 50

#model.load_weights(models_dir+modelname)

if not os.path.exists(os.path.dirname('models/'+modelfilename+"/")):
    try:
        os.makedirs(os.path.dirname('models/'+modelfilename+"/"))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

with open('models/'+modelfilename+"/summary.txt", "w") as text_file:
    model.summary(print_fn=lambda x: text_file.write(x + '\n'))
    text_file.write("\nepochs: {}\nsteps_per_epoch: {}\ncheckpoints: {}".format(epochs, steps_per_epoch, checkpoints))

    n = 0

    while n < epochs:
        print("\nEpoch {}:".format(n))
        history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=checkpoints,
        validation_data=validation_generator,
        validation_steps=int(steps_per_epoch/10))
        n += checkpoints

        #scores = model.evaluate(data[2], data[3], batch_size=batch_size)
        #print("Scores on test set for run {}: loss/accuracy={}".format(n, tuple(scores)))
        model.save('models/'+modelfilename+'/checkpoint_{}.h5'.format(n))
        text_file.write("\nEpoch {}: {}".format(n, history.history))

        #No need to keep after 0.95, abort
        if history.history["acc"][-1] >= 0.95:
            print('Achieved 95% acc')
            n = epochs

        """if n+checkpoints > epochs:
            checkpoints = epochs - n"""



"""image = validation_generator[0][0][2:3]
c
plot.imshow(image)
plot.show()

#####################################

import innvestigate
import innvestigate.utils.visualizations as ivis
import innvestigate.utils as iutils

methods = [("lrp.sequential_preset_a_flat",    {"epsilon": 1}, ivis.heatmap,    "LRP-PresetAFlat")]
method = methods[0]

model_wo_sm = iutils.keras.graph.model_wo_softmax(model)
analyzer = innvestigate.create_analyzer(method[0], model_wo_sm, **method[1])#, neuron_selection_mode="index")
analysis = analyzer.analyze(image)
processed_analysis = iutils.postprocess_images(analysis.copy())
processed_analysis = method[2](processed_analysis)

print (model.predict(image), model.predict_classes(image))

a, b = processed_analysis.max(), processed_analysis.min()
c, d = 0, 1

def preprocessing(X):
        # shift original data to [0, b-a] (and copy)
        X = X - a
        # scale to new range gap [0, d-c]
        X /= (b-a)
        X *= (d-c)
        # shift to desired output range
        X += c
        return X


plot.imshow(preprocessing(processed_analysis).squeeze(), cmap='seismic', interpolation='nearest')
plot.show()"""