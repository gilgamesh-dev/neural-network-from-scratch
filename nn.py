
import math
import random

# X_data is your inputs,
# Y_data is your expected outputs.
# Each row is one training example.
x_data = [ [0,0], [0,1], [1,0], [1,1] ]
y_data = [[0], [1], [1], [0]]

def sigmoid(x: float) -> float:
    return 1 / ( 1 + math.exp(-x) )

def dot(a: list[float], b: list[float]) -> float:
    # Takes two list of numbers, and returns a single number
    # multiply matching pairs, then sum
    # Example: dot([2,3], [4,5]) -> (2 x 4) + ( 3 x 5 ) = 8 + 15 = 23
    total = 0.0
    for i in range(len(a)):
        # we get the inputs and multiply it by the weights 
        total += a[i] * b[i]
    return total

def neuron(inputs: list[float], weights: list[float], bias: float) -> float:
    z = dot(inputs, weights) + bias
    return sigmoid(z)


def layer(inputs: list[float], weights: list[list[float]], biases: list[float]) -> list[float]:
    outputs: list[float] = []
    for i in range(len(weights)):
        result = neuron(inputs, weights[i], biases[i])
        outputs.append(result)
    return outputs


def forward(inputs: list[float], network_weights: list[list[list[float]]], network_biases: list[list[float]]) -> list[float]:
    # Push input through each layer — output of one becomes input to the next
    current = inputs
    for i in range(len(network_weights)):
        current = layer(current, network_weights[i], network_biases[i])
    return current


# ============================================================
# STEP 2: LOSS FUNCTION — how wrong is our prediction?
# ============================================================

def mse(prediction: list[float], target: list[float]) -> float:
    # Mean Squared Error: average of (prediction - target)² for each output
    # If prediction is close to target → small loss (good)
    # If prediction is far from target → big loss (bad)
    total = 0.0
    for i in range(len(prediction)):
        total += (prediction[i] - target[i]) ** 2
    return total / len(prediction)


# ============================================================
# STEP 3: BACKPROPAGATION — which weights caused the error?
# This is where calculus (chain rule) kicks in.
# ============================================================

def sigmoid_derivative(a: float) -> float:
    # Derivative of sigmoid, given the sigmoid OUTPUT (not input)
    # From your calculus note: σ'(x) = σ(x) × (1 - σ(x))
    # Since 'a' is already sigmoid(x), it's just a × (1 - a)
    return a * (1 - a)


def forward_with_cache(inputs: list[float], network_weights: list[list[list[float]]], network_biases: list[list[float]]) -> tuple[list[float], list[list[float]]]:
    # Same as forward, but saves every layer's output
    # Backprop needs these saved values to compute derivatives
    current = inputs
    activations: list[list[float]] = [inputs]  # save input as "layer 0"

    for i in range(len(network_weights)):
        current = layer(current, network_weights[i], network_biases[i])
        activations.append(current)

    return current, activations


def backward(target: list[float], network_weights: list[list[list[float]]], activations: list[list[float]]) -> tuple[list[list[list[float]]], list[list[float]]]:
    # Walk backwards through the network, computing gradients using the chain rule
    # Returns: gradient for every weight and every bias
    num_layers = len(network_weights)
    grad_w: list[list[list[float]]] = [[] for _ in range(num_layers)]
    grad_b: list[list[float]] = [[] for _ in range(num_layers)]

    # Start at the output: dL/dA = 2 * (prediction - target) / n
    # This is the derivative of MSE (from your calculus note)
    output = activations[-1]
    dA: list[float] = [2 * (output[i] - target[i]) / len(target) for i in range(len(target))]

    # Walk backwards through each layer
    for l in range(num_layers - 1, -1, -1):
        A = activations[l + 1]       # this layer's output
        A_prev = activations[l]      # this layer's input (previous layer's output)

        # dZ = dA * sigmoid'(A)  — chain rule: multiply loss gradient by activation derivative
        dZ: list[float] = [dA[i] * sigmoid_derivative(A[i]) for i in range(len(A))]

        # dW = dZ × A_prev  — how much each weight contributed to the error
        dW: list[list[float]] = []
        for i in range(len(dZ)):
            dW.append([dZ[i] * A_prev[j] for j in range(len(A_prev))])
        grad_w[l] = dW

        # db = dZ  — bias gradient is just dZ (because ∂z/∂b = 1)
        grad_b[l] = [dz for dz in dZ]

        # Propagate error to the previous layer: dA_prev = Wᵀ · dZ
        # This is the transpose from your linear algebra note
        if l > 0:
            dA = [0.0] * len(A_prev)
            for j in range(len(A_prev)):
                for i in range(len(dZ)):
                    dA[j] += network_weights[l][i][j] * dZ[i]

    return grad_w, grad_b


# ============================================================
# STEP 4: GRADIENT DESCENT — nudge weights to reduce error
# ============================================================

def update(weights: list[list[list[float]]], biases: list[list[float]], grad_w: list[list[list[float]]], grad_b: list[list[float]], learning_rate: float) -> None:
    # For every weight: weight = weight - learning_rate × gradient
    # The gradient points "uphill" (toward more error)
    # Subtracting it moves us "downhill" (toward less error)
    for l in range(len(weights)):
        for i in range(len(weights[l])):
            for j in range(len(weights[l][i])):
                weights[l][i][j] -= learning_rate * grad_w[l][i][j]
            biases[l][i] -= learning_rate * grad_b[l][i]


# ============================================================
# STEP 5: CREATE NETWORK — random starting weights
# ============================================================

def create_network(layer_sizes: list[int]) -> tuple[list[list[list[float]]], list[list[float]]]:
    # layer_sizes example: [2, 4, 1] means 2 inputs, 4 hidden neurons, 1 output
    # Returns random weights and zero biases for every layer
    weights: list[list[list[float]]] = []
    biases: list[list[float]] = []

    for i in range(len(layer_sizes) - 1):
        n_in = layer_sizes[i]      # neurons in previous layer
        n_out = layer_sizes[i + 1] # neurons in this layer

        # Xavier initialization: scale by sqrt(1/n_in)
        # Prevents signals from getting too big or too small
        scale = math.sqrt(1.0 / n_in)

        # Each neuron gets n_in random weights
        layer_w: list[list[float]] = []
        for _ in range(n_out):
            neuron_w: list[float] = [random.gauss(0, scale) for _ in range(n_in)]
            layer_w.append(neuron_w)
        weights.append(layer_w)

        # Each neuron gets one bias, starting at 0
        layer_b: list[float] = [0.0 for _ in range(n_out)]
        biases.append(layer_b)

    return weights, biases


# ============================================================
# STEP 6: TRAINING LOOP — put it all together
# ============================================================

def train(x_data: list[list[float]], y_data: list[list[float]], layer_sizes: list[int], learning_rate: float = 2.0, epochs: int = 10000) -> tuple[list[list[list[float]]], list[list[float]]]:
    random.seed(42)
    weights, biases = create_network(layer_sizes)

    for epoch in range(epochs):
        total_loss = 0.0

        # For each training example:
        for i in range(len(x_data)):
            X = x_data[i]
            Y = y_data[i]

            # 1. Forward pass — get prediction (and save values for backprop)
            prediction, activations = forward_with_cache(X, weights, biases)

            # 2. Compute loss — how wrong are we?
            loss = mse(prediction, Y)
            total_loss += loss

            # 3. Backward pass — compute gradients (chain rule)
            grad_w, grad_b = backward(Y, weights, activations)

            # 4. Update weights — gradient descent (nudge downhill)
            update(weights, biases, grad_w, grad_b, learning_rate)

        # Print progress every 1000 epochs
        if epoch % 1000 == 0:
            avg_loss = total_loss / len(x_data)
            print(f"Epoch {epoch:5d} | Loss: {avg_loss:.6f}")

    return weights, biases


# ============================================================
# RUN IT
# ============================================================

# Train: 2 inputs → 4 hidden neurons → 1 output
weights, biases = train(x_data, y_data, layer_sizes=[2, 4, 1])

# Test — see if it learned XOR
print("\n--- Results ---")
for i in range(len(x_data)):
    output = forward(x_data[i], weights, biases)
    print(f"Input: {x_data[i]}  Predicted: {output[0]:.4f}  Target: {y_data[i][0]}")
