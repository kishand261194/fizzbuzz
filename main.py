
# ## Logic Based FizzBuzz Function [Software 1.0]
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard, Callback
import numpy as np
import matplotlib.pyplot as plt

# class TrainingPlot(Callback):
#
#     # This function is called when the training begins
#     def on_train_begin(self, logs={}):
#         # Initialize the lists for holding the logs, losses and accuracies
#         self.losses = []
#         self.acc = []
#         self.val_losses = []
#         self.val_acc = []
#         self.logs = []
#
#     # This function is called at the end of each epoch
#     def on_epoch_end(self, epoch, logs={}):
#
#         # Append the logs, losses and accuracies to the lists
#         self.logs.append(logs)
#         self.losses.append(logs.get('loss'))
#         self.acc.append(logs.get('acc'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.val_acc.append(logs.get('val_acc'))
#
#         # Before plotting ensure at least 2 epochs have passed
#         if len(self.losses) > 1:
#
#             N = np.arange(0, len(self.losses))
#
#             # You can chose the style of your preference
#             # print(plt.style.available) to see the available options
#             #plt.style.use("seaborn")
#
#             # Plot train loss, train acc, val loss and val acc against epochs passed
#             plt.figure()
#             plt.plot(N, self.losses, label = "train_loss")
#             plt.plot(N, self.acc, label = "train_acc")
#             plt.plot(N, self.val_losses, label = "val_loss")
#             plt.plot(N, self.val_acc, label = "val_acc")
#             plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
#             plt.xlabel("Epoch #")
#             plt.ylabel("Loss/Accuracy")
#             plt.legend()
#             # Make sure there exists a folder called output in the current directory
#             # or replace 'output' with whatever direcory you want to put in the plots
#             plt.savefig('output/Epoch-{}.png'.format(epoch))
#             plt.close()

def fizzbuzz(n):

    # Logic Explanation
    if n % 3 == 0 and n % 5 == 0:
        return 'fizzbuzz'
    elif n % 3 == 0:
        return 'fizz'
    elif n % 5 == 0:
        return 'buzz'
    else:
        return 'other'

def encodeData(data):

    processedData = []

    for dataInstance in data:

        # Why do we have number 10?
        # Here we are encoding the numbers to their binary form and we know 2^10 = 1024 which means to represent all
        # numbers till 1024 in binary, we will be needing 10 bits. Our training + testing data set is within 1-1000
        # which is less than 1024. Therefore we will be needing 10 bits to represent the numbers.
        #
        #
        # We are right shifting each number(10 times) and performing an and-operation with 1
        # each time to extract the right most bit
        processedData.append([dataInstance >> d & 1 for d in range(10)])

    return np.array(processedData)

def encodeLabel(labels):

    processedLabel = []

    for labelInstance in labels:
        if(labelInstance == "fizzbuzz"):
            # Fizzbuzz
            processedLabel.append([3])
        elif(labelInstance == "fizz"):
            # Fizz
            processedLabel.append([1])
        elif(labelInstance == "buzz"):
            # Buzz
            processedLabel.append([2])
        else:
            # Other
            processedLabel.append([0])
    return np_utils.to_categorical(np.array(processedLabel),4)  # to_categorical takes in vector and coverts them to
                                                                # binary class matrix (fizzbuzz, fizz, buzz, others),
                                                                # row = processedLabel (0 to number of class)
                                                                # column = class
                                                                # eg. [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] ==
                                                                #     [0,1,2,3]
                                                                #     [Others, Buzz, Fizz, FizzBuzz]


# ## Model Definition
input_size = 10

# Why dropout?
# Eg. In a house of 4, only 1 person knows to cook and cooks everyday. And other 3 where always dependent
# on the first guy to cook food. The first guy goes on a trip for a week, No food in the house.
# Now the other 3 had to learn cooking for survival. Hence, Got more balance in the house.
#
# drop out is used to randomly ignore few neurons during the training period,
# this makes the other neurons to step up and make up for the missing neurons and predict. Hence, the neural
# network as a whole wont be very sensitive to specific weights of neurons. Achieving higher generalization.
drop_out = 0.31 # when 30% - acc 83%, 20% - acc 71%, acc 35% - 79%, acc 31% - 85%, acc 32% - 84%

first_dense_layer_nodes  = 256 #256
second_dense_layer_nodes = 256 #when 10 accuracy drops to 85
third_dense_layer_nodes = 4

def get_model():

    # Why do we need a model?
    # We cant just look at the data and start making prediction, why ? we might not have data for all possible
    # combination, this calls for the need to have models,
    # which takes in data, analyzes and learns patterns from the data.
    # Trys to predict when a new unseen  data comes up.
    #
    #
    # Why use Dense layer and then activation?
    # Dense layer gives the input signal vector(Xi) and associated weights(W) which are needed by
    # the activation function to produce the desired output. Therefore activation has to be followed only after
    # dense layer.
    #
    #
    #
    # Why use sequential model with layers?
    # Because we have single input, single output and layers are stacked one over the other and traversed in order
    # and not re-used. All these make sequential model most suitable for FizzBuzz.


    model = Sequential()
#    a=['relu', 'tanh', 'sigmoid', 'linear']
#    b=['rmsprop', 'adam', 'sgd', 'nadam']
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('relu')) #while using tanh accuracy is constant
    model.add(Dropout(drop_out))

    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('relu'))
    model.add(Dropout(drop_out))

    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax'))
    # Why Softmax?
    # in the output layer the four neurons will be having any real number value and not the probabilty
    # of it belonging to a class. What softmax does is it brings all these values to add to 1, which means,
    # it gives the probablity of the input being in each class
    model.summary()

    # Why use categorical_crossentropy?
    # Beause it is a classication problem with multiple classes.
    # Is good to for classication. Higher Learning rate, you will reach minimize loss
    #rmsprop=RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
    adam=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def decodeLabel(encodedLabel):
    if encodedLabel == 0:
        return "other"
    elif encodedLabel == 1:
        return "fizz"
    elif encodedLabel == 2:
        return "buzz"
    elif encodedLabel == 3:
        return "fizzbuzz"

### Create Training and Testing Datasets in CSV Format

def createInputCSV(start,end,filename):

    # Why list in Python?
    # list the python verision of saving data in sequence, as we need input and output to be
    # in sequence we are using list
    inputData   = []
    outputData  = []

    # Why do we need training Data?
    # Training data is the source from which patterns and features needed for
    # prediction are extracted from. Training data is like the books we read before writing an exam.
    for i in range(start,end):
        inputData.append(i)
        outputData.append(fizzbuzz(i))

    # Why Dataframe?
    # Reason for using here:
    # It provides a direct method to convert the 2d dataframe into CSV (easiest way to extact Traning, Testing data set)
    # In general:
    # It has a huge number of built-in methods and attributes (abstracts out many common functions),
    # which lets us focus on the needed things. Eg. T - Transpose, dot - matrix multiplication, join and etc
    dataset = {}
    dataset["input"]  = inputData
    dataset["label"] = outputData

    # Writing to csv
    pd.DataFrame(dataset).to_csv(filename)

    print(filename, "Created!")


# ## Processing Input and Label Data

def processData(dataset):

    # Why do we have to process?
    # By processing the data we convert the input into a 10 bit binary, which gives us with more
    # number of nodes to play with, which makes it easy for the system to process and predict better
    data   = dataset['input'].values
    labels = dataset['label'].values

    processedData  = encodeData(data)
    processedLabel = encodeLabel(labels)

    return processedData, processedLabel


# Create datafiles
createInputCSV(101,1001,'training.csv')
createInputCSV(1,101,'testing.csv')

validation_data_split = 0.2
num_epochs = 700 # is one full pass of the all the samples that we have, eg. if the batch_size=10
                    # total dataset(sample) we have is 10000, to complete one epoch it will take 1000 batches
model_batch_size = 128 #128 # batch size is number of sample that we pass to the model at one time, mainly depends on the
                          # computation power of the system
tb_batch_size = 32 #32
early_patience = 100
#acc=[]
#others={}
#plot_losses = TrainingPlot()
#for f in range(4):
#a=['relu', 'tanh', 'sigmoid', 'linear']
#    b=['rmsprop', 'adam', 'sgd', 'nadam']
#    for h in range(4):
model = get_model()
tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)

        # Early stopping is a way to stop training when your loss starts to increase (decrease in validation),
        # here we monitor is  validation loss
        # 'patience' argument is the number of epochs before stopping, once your loss starts to increase (stops improving)
        # 'mode' is to tell keras to look for change in increase or decrease , if min is given it observes for decrease
earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')

        # Read Dataset
dataset = pd.read_csv('training.csv')

        # Process Dataset
processedData, processedLabel = processData(dataset)
history = model.fit(processedData
                    , processedLabel
                    , validation_split=validation_data_split
                    , epochs=num_epochs
                    , batch_size=model_batch_size
                    , callbacks = [tensorboard_cb, earlystopping_cb]) # can add plot_losses callback

training_loss = history.history['loss']
test_loss = history.history['val_loss']
training_acc = history.history['acc']
test_acc = history.history['val_acc']
    #Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show();

plt.plot(epoch_count, training_acc, 'r--')
plt.plot(epoch_count, test_acc, 'b-')
plt.legend(['Training Acc', 'Test Acc'])
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.show();
#others[(a[f], b[h])]= history.history['val_acc'].pop()
wrong   = 0
right   = 0

testData = pd.read_csv('testing.csv')

processedTestData  = encodeData(testData['input'].values)
processedTestLabel = encodeLabel(testData['label'].values)
predictedTestLabel = []

for i,j in zip(processedTestData,processedTestLabel):
    y = model.predict(np.array(i).reshape(-1,10))
    predictedTestLabel.append(decodeLabel(y.argmax()))

    if j.argmax() == y.argmax():
        right = right + 1
    else:
        wrong = wrong + 1

print("Errors: " + str(wrong), " Correct :" + str(right))

print("Testing Accuracy: " + str(right/(right+wrong)*100))
#acc.append(right/(right+wrong)*100)
# Please input your UBID and personNumber
testDataInput = testData['input'].tolist()
testDataLabel = testData['label'].tolist()

testDataInput.insert(0, "UBID")
testDataLabel.insert(0, "kishandh")

testDataInput.insert(1, "personNumber")
testDataLabel.insert(1, "50287619")

predictedTestLabel.insert(0, "")
predictedTestLabel.insert(1, "")

output = {}
output["input"] = testDataInput
output["label"] = testDataLabel

output["predicted_label"] = predictedTestLabel

opdf = pd.DataFrame(output)
opdf.to_csv('output.csv')
#plt.legend()
#plt.xlabel('others')
#plt.ylabel('acc')
#for xy in zip(others, acc):
#    plt.annotate('(%s, %s)' % xy, xy=xy, textcoords='data')
        #plt.annotate('%0.2f' % max(var), xy=(1, max(var)), xytext=(8, 0),
        #             xycoords=('axes fraction', 'data'), textcoords='offset points')
#plt.show();
