import numpy as np 
import h5py
import cv2
import os
from glob import glob


#////////////////////////testing some images//////////////////////////
def test():
	index = 0
	for i in range(10):
		cv2.imshow("image",trainX[i])
		cv2.waitKey(1000)
		cv2.destroyAllWindows()
		print(trainy[i])


# n = 0 to n and arr of shape(x,)
def Y_generator(n,arr):
	x = arr.shape[0]
	Y = np.zeros([n+1,x])
	for i in range(x):
		num = arr[i]
		Y[num][i] = 1

	return Y

# test = np.array([0,1,2,3])
# print(Y_generator(3,test))


#/////////////// start writing neural network functions //////////////////////////////
#retrieving parameters dictionary





def initialize_parameters(layers_dim):
	parameters = {}

	# for i in range(1,len(layers_dim)):
	# 	ni = layers_dim[i-1]
	# 	ni_next = layers_dim[i]
	# 	parameters["W" + str(i)] = np.random.randn(ni_next,ni)*0.01
	# 	parameters["b" + str(i)] = np.zeros([ni_next,1],dtype=int)
	with h5py.File("parametersnew.h5","r")as hf:
		for l in range(3):
			nameW = "W" + str(l+1)
			nameb = "b" + str(l+1)
			parameters[nameW] = hf[nameW][:]
			parameters[nameb] = hf[nameb][:]


	return parameters


def linear_forward(A,W,b):
	cache = (A,W,b)
	Z = np.matmul(W,A) + b
	return Z,cache

def relu(Z):
	mask = (Z > 0)
	A = Z*mask
	return A,Z

def sigmoid(Z):
	A = 1/(1+ np.exp(-Z))
	return A,Z


def softmax(Z):
	temp = np.exp(Z)
	temp2 = temp.sum(axis=0)
	A = temp/temp2
	return A,Z

def linear_activation_forward(A_prev,W,b,activation):
	Z,lin_cache = linear_forward(A_prev,W,b)

	if(activation == "relu"):
		A,act_cache = relu(Z)
		

	elif(activation == "sigmoid"):
		A,act_cache = sigmoid(Z)

	elif(activation == "softmax"):
		A,act_cache = softmax(Z)

	cache = (lin_cache,act_cache)
	return A,cache


def L_model_forward(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        ### START CODE HERE ### (≈ 2 lines of code)
        Wl = parameters["W" + str(l)]
        Bl = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev,Wl,Bl,"relu")
        caches.append(cache)
        ### END CODE HERE ###
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    ### START CODE HERE ### (≈ 2 lines of code)
    WL = parameters["W" + str(L)]
    BL = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A,WL,BL,"softmax")
    caches.append(cache)
    ### END CODE HERE ###
    
    return AL, caches


def compute_cost(AL, Y):
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = np.sum(Y*np.log(AL))*(-1/m)
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost



def linear_backward(dZ, cache):
    
    A_prev, W, b = cache
    m = A_prev.shape[1]

    ### START CODE HERE ### (≈ 3 lines of code)
    dW = np.matmul(dZ,A_prev.T)*(1/m)
    db = np.sum(dZ,axis=1,keepdims=True)*(1/m)
    dA_prev = np.matmul(W.T,dZ)
    ### END CODE HERE ###
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
  
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = dA*(activation_cache >= 0)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        ### END CODE HERE ###
        
    elif activation == "sigmoid":
        ### START CODE HERE ### (≈ 2 lines of code)
        dZ = dA*(sigmoid(activation_cache)[0]*(1-sigmoid(activation_cache)[0]))
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
        ### END CODE HERE ###
    elif activation == "softmax":
    	dZ = softmax(activation_cache)[0] - dA  # Y_hat - Y is derivative wrt Z in softmax
    	dA_prev, dW, db = linear_backward(dZ,linear_cache)
    
    return dA_prev, dW, db



def L_model_backward(AL, Y, caches):
   
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    ### START CODE HERE ### (1 line of code)
    dAL = Y

    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,current_cache,"softmax")
    ### END CODE HERE ###
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        ### START CODE HERE ### (approx. 5 lines)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
        ### END CODE HERE ###

    return grads


def update_parameters(parameters, grads, learning_rate):
  
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    ### START CODE HERE ### (≈ 3 lines of code)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    ### END CODE HERE ###
    return parameters



def L_layer_model(X, Y, layers_dims, learning_rate = 0.07, num_iterations = 3000, print_cost=False):#lr was 0.009

    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    ### START CODE HERE ###
    parameters = initialize_parameters(layers_dims)
    ### END CODE HERE ###
    
    adam= {}


    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        ### START CODE HERE ### (≈ 1 line of code)
        AL, caches = L_model_forward(X,parameters)
        ### END CODE HERE ###
        
        # Compute cost.
        ### START CODE HERE ### (≈ 1 line of code)
        cost = compute_cost(AL,Y)
        ### END CODE HERE ###
    
        # Backward propagation.
        ### START CODE HERE ### (≈ 1 line of code)
        grads = L_model_backward(AL,Y,caches)
        ### END CODE HERE ###
 
        # Update parameters.
        ### START CODE HERE ### (≈ 1 line of code)


        parameters = update_parameters(parameters,grads,learning_rate)
        ### END CODE HERE ###
                
        # Print the cost every 100 training example
        if print_cost and i % 30 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # # plot the cost
    # plt.plot(np.squeeze(costs))
    # plt.ylabel('cost')
    # plt.xlabel('iterations (per hundreds)')
    # plt.title("Learning rate =" + str(learning_rate))
    # plt.show()
    
    return parameters



# parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)



#///////////////////////////////////////////////training session//////////////////////////////////////////
#////////////////////@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@///////////

# load dataset
path = os.path.abspath('..')
path1 = os.path.join(path,"dataset","train.hdf5")
path2 = os.path.join(path,"dataset","test.hdf5") 

##loading training set
with h5py.File(path1,"r") as hf:
	trainX= hf["image"][:]
	trainy = hf["label"][:]


## loading test set
with h5py.File(path2,"r") as hf:
	testX= hf["image"][:]
	testy = hf["label"][:]


# trainy1 = Y_generator(9,trainy)
# trainX = trainX.reshape(trainX.shape[0],-1)/255
# trainX = trainX.T

#define layers dimension
layers_dims = [784,16,16,10]


# parameters = L_layer_model(trainX, trainy1, layers_dims, num_iterations = 1500, print_cost = True)


# with h5py.File("parameters.h5","a")as hf:
# 	for l in range(len(parameters)//2):
# 		nameW = "W" + str(l+1)
# 		nameb = "b" + str(l+1)
# 		Wl = parameters[nameW]
# 		bl = parameters[nameb]
# 		hf.create_dataset(nameW,data=Wl)
# 		hf.create_dataset(nameb,data=bl)


#///////////////////////////////////////////////////////Testing session////////////////////////////////
#now we got values of parameters its testing time




# print(testX.shape)





# AL_train,caches = L_model_forward(trainX,parameters)
# AL_test,caches = L_model_forward(testX,parameters)

# ans_train = AL_train.argmax(axis=0)
# ans_test = AL_test.argmax(axis=0)

# #train accuracy
# accuracy_train = np.sum(ans_train == trainy)/len(trainy)
# accuracy_test = np.sum(ans_test == testy)/len(testy)

# print("train accuracy ",accuracy_train)
# print("test accuracy ", accuracy_test)

# got 95 % accuracy //////////////////////////////////


parameters = {}

with h5py.File("parameters.h5","r")as hf:
	for l in range(3):
		nameW = "W" + str(l+1)
		nameb = "b" + str(l+1)
		parameters[nameW] = hf[nameW][:]
		parameters[nameb] = hf[nameb][:]



def make_image_path_list():
	#abs_path of current directory
	PATH = os.path.abspath('.')

	#join the test and train folders path
	Source= os.path.join(PATH,'images')
	print(Source)
	#search for png files in folder then sort the images according to their names
	images = glob(os.path.join(Source,"*.jpeg"))
	print(images)
	return images

images = make_image_path_list()



#///////////////////////////////// testing your images //////////////////////////

for i in images:
	filename = i
	img = cv2.imread(filename)
	gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gray_resized = cv2.resize(gray_img,(28,28),interpolation=cv2.INTER_AREA)

	gray_resized = ~gray_resized


	ret,gray_resized = cv2.threshold(gray_resized,127,255,cv2.THRESH_BINARY)

	kernel = np.ones((2,2), np.uint8) 

	gray_resized = cv2.dilate(gray_resized, kernel, iterations=1) 
	# print(gray_resized)
	X = gray_resized.reshape(-1)
	X = X.reshape(len(X),1)
	X = X/255

	AL,caches = L_model_forward(X,parameters)
	# print(AL)
	print(AL.argmax(axis=0))

	cv2.imshow("full",gray_img)
	cv2.imshow("resized",gray_resized)
	cv2.waitKey(4000)
	cv2.destroyAllWindows()

	



# gray_resized = testX[1]
# print(gray_resized)

# cv2.imshow("resized",gray_resized)
# cv2.waitKey(5000)
# cv2.destroyAllWindows()

# X = gray_resized.reshape(-1)
# X = X.reshape(len(X),1)
# X = X/255


# AL,caches = L_model_forward(X,parameters)
# print(AL)
# print(AL.argmax(axis=0))



