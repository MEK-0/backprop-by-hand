"""
=============================================================
            Neural Network from Scratch
=============================================================
Architecture : MLP -Multiple Layer Perceptron-
Struct : 784 → 128 → ReLU → 64 → ReLU → 10 → LogSoftmax
=============================================================
"""
import time
import numpy as np 
from tqdm import tqdm
from scipy.special import logsumexp
from keras.datasets.mnist import load_data

#============================================================
# MLP Layer (Full Connected / Linear Layer )
# Each Layer holds weights W and bias b.
# Forward : Z = X @ W.T + b
# Backward : compute gradients for W, b, and input X
#============================================================

class MLP():

    def __init__(self, din, dout):
        # Xavier /Glorot initialization
        #Keeps gradients from vanishing or exploding at the start
        scale = np.sqrt(6) / np.sqrt(din + dout)
        self.W = (2 * np.random.rand(dout, din) - 1 ) * scale # shape: (dout, din)
        self.b = (2 * np.random.rand(dout) - 1) * scale # shape: (dout,)


    def forward(self,x):
        # x shape: (batch_size , din)
        # Save x for backward pass
        self.x = x
        return x @ self.W.T + self.b #shape: (batch_size, dout)


    def backward(self,gradout):
        # gradout shape: (bacth_size, dout)
        self.deltaW = gradout.T @ self.x # gradient for w
        self.deltab = gradout.sum(0) # gradient for b
        return gradout @ self.W # gradient for input X
    

#============================================================
# ReLU Activation Function
# Introduces non-linearity into the network.
# Forward : f(x) = max(0, x)
# Backward : pass gradient where x > 0, zero out where x <= 0
#============================================================

class ReLU():

    def forward(self,x):
        # Save x to use in backword pass
        self.x = x
        return np.maximum(0,x)


    def backward(self,gradout):
        # Copy gradient and kill values where input was negative
        grad = gradout.copy()
        grad[self.x < 0] = 0.
        return grad

#============================================================
# LogSoftmax Activation Function
# Converts raw scores (logits) to log-probabilities.
# More numerically stable than plain Softmax + log.
# Forward : log_softmax(x) = x - log(sum(exp(x)))
# Backward : Jacobian of log_softmax = I - softmax(x)
#============================================================

class LogSoftmax():

    def forward(self,x):
        self.x = x
        return x - logsumexp(x, axis=1)[..., None]

    def backward(self, gradout):
        #Build Jacobian: shape (batch_size, dout, dout)
        # Eye matrix repeated for eacg sample in the batch
        jacobian = np.eye(self.x.shape[1])[None, ...]

        #Subtract softmax: jacobian = I - softmax(x)
        softmax = np.exp(self.x) / np.sum(np.exp(self.x), axis = 1, keepdims= True)
        jacobian = jacobian - softmax[..., None]

        return (np.matmul(jacobian, gradout[..., None]))[:,:, 0]


# =============================================================
#  Negative Log Likelihood Loss (NLLLoss)
# Works together with LogSoftmax to compute cross-entropy loss.
# Forward : loss = -sum(log_prob[correct_class]) for each sample
# Backward: gradient is -1 at correct class index, 0 elsewhere
# =============================================================

class NLLLoss():

    def forward(self, pred, true):
        # pred shape : (batch_size, num_classes) — log-probabilities
        # true shape : (batch_size,)              — correct class indices
        self.pred = pred
        self.true = true

        loss = 0
        for b in range(pred.shape[0]):
            loss -= pred[b, true[b]]  # pick log-prob of correct class
        return loss

    def backward(self):
        # Jacobian: -1 where prediction was correct class, 0 elsewhere
        jacobian = np.zeros((self.pred.shape[0], self.pred.shape[1]))
        for b in range(self.pred.shape[0]):
            jacobian[b, self.true[b]] = -1
        return jacobian  # shape: (batch_size, num_classes)

    def __call__(self, pred, true):
        # Allows using the object like a function: loss(pred, true)
        return self.forward(pred, true)

# =============================================================
# SequentialNN — Container for all layers
# Chains layers together for forward and backward passes.
# Forward : left to right  (input → output)
# Backward: right to left  (output → input), chain rule
# =============================================================

class SequentialNN():

    def __init__(self, blocks: list):
        # blocks: list of layers e.g. [MLP, ReLU, MLP, LogSoftmax]
        self.blocks = blocks

    def forward(self, x):
        # Pass input through each layer in order
        for block in self.blocks:
            x = block.forward(x)
        return x

    def backward(self, gradout):
        # Pass gradient through each layer in REVERSE order
        for block in self.blocks[::-1]:
            gradout = block.backward(gradout)
        return gradout

# =============================================================
#  Optimizer (Stochastic Gradient Descent)
# Updates weights of MLP layers using computed gradients.
# Rule: W = W - lr * deltaW
#        b = b - lr * deltab
# Only MLP layers have learnable parameters (W and b).
# =============================================================

class Optimizer():

    def __init__(self, lr, compound_nn: SequentialNN):
        # lr          : learning rate (step size)
        # compound_nn : the full network whose weights will be updated
        self.lr = lr
        self.compound_nn = compound_nn

    def step(self):
        # Loop through all blocks, update only MLP layers
        for block in self.compound_nn.blocks:
            if block.__class__ == MLP:
                block.W = block.W - self.lr * block.deltaW
                block.b = block.b - self.lr * block.deltab

# =============================================================
# STEP 8: Training Loop
# Runs forward pass, computes loss, runs backward pass,
# and updates weights for a given number of epochs.
# Uses mini-batch SGD: random subset of data each epoch.
# =============================================================

def train(model, optimizer, trainX, trainy,
          loss_fct=NLLLoss(), nb_epochs=14000, batch_size=100):

    training_loss = []

    for epoch in tqdm(range(nb_epochs)):

        # --- 1. Sample a random mini-batch ---
        batch_idx = [np.random.randint(0, trainX.shape[0]) for _ in range(batch_size)]
        x      = trainX[batch_idx]   # shape: (batch_size, 784)
        target = trainy[batch_idx]   # shape: (batch_size,)

        # --- 2. Forward pass ---
        prediction = model.forward(x)

        # --- 3. Compute loss ---
        loss_value = loss_fct(prediction, target)
        training_loss.append(loss_value)

        # --- 4. Backward pass ---
        gradout = loss_fct.backward()
        model.backward(gradout)

        # --- 5. Update weights ---
        optimizer.step()

    return training_loss

if __name__ == "__main__":
    # --- 1. Load and Preprocess Dataset ---
    print("Loading MNIST dataset...")
    (X_train, y_train), (X_test, y_test) = load_data()

    # Flatten images from 2D (28x28) to 1D (784)
    # Normalize pixel values to the range [0, 1] by dividing by 255.0
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # --- 2. Define Model Architecture ---
    # Struct : 784 -> 128 -> ReLU -> 64 -> ReLU -> 10 -> LogSoftmax
    model = SequentialNN([
        MLP(784, 128),
        ReLU(),
        MLP(128, 64),
        ReLU(),
        MLP(64, 10),
        LogSoftmax()
    ])

    # --- 3. Optimizer and Training ---
    # Define the learning rate
    learning_rate = 0.01 
    optimizer = Optimizer(lr=learning_rate, compound_nn=model)

    print("Training started...")
    # You can decrease nb_epochs (e.g., to 2000) to shorten training time
    start_time = time.time()
    loss_history = train(model, optimizer, X_train, y_train, nb_epochs=5000, batch_size=128)
    end_time =time.time()
    elapsed_time = end_time - start_time
    # --- 4. Evaluate on Test Data ---
    print("\nEvaluating on test data...")
    # Perform a forward pass with the test data
    test_preds = model.forward(X_test)
    
    # Select the class with the highest log-probability as the prediction
    predicted_classes = np.argmax(test_preds, axis=1)
    
    # Calculate Accuracy
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"\n Total learning time: {elapsed_time:.2f} second ({elapsed_time / 60:.2f}) minute")
    
    














