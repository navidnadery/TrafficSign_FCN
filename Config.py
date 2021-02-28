shuffle = True # Whether shuffle the data
batch_size = 8 # batch_size for train and validation
seed = 1234 # Random seed
epochs = 5 # Number of epochs for training

input_size = 100 # network input size (image size)
input_chan = 3 # number of input channels (3 for RGB and 1 for black and white)
cnn_dim = [32,64,128] # Number of CNN filters (Neurons)
cnn_kernel = [11,9,5] # the Size of CNN kernels (filter size)
cnn_pool = [1,3] # the pooling size of the Max Pooling layers
dr = 0.0 # dropout Rate

reg = 0 #.5 # The weight of regularization term
learning_rate = 5e-1 # The Learning Rate

checkpoint_path = '.' # path to save checkpoints to
Train_path = 'Training' # path for training data where each class samples are placed in the corresponding sub-dir
Test_path = 'Test' # path for test data where each class samples are placed in the corresponding sub-dir

ext = 'ppm' # the extension of samples (jpg, png, etc.)