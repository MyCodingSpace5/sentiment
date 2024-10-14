import math

def attention(query: int, key: int, value: int, weights: int):
    return (query * weights) + (key * weights) + (value * weights)

def attention_backpropgation(output: int, network_output: int, weights: int, learning_rate: float):
    delta = output - network_output
    weights = weights - ((delta / output) * learning_rate)
    return weights

def softmax(x: int, features: list) -> int:
    exp_values = [math.exp(f) for f in features]
    return math.exp(x) / sum(exp_values)

def cross_entropy(output: int, feature: int) -> float:
    return -math.log(output - feature, math.e)

def sigmoid(x: int):
    return 1 / (1 + math.exp(-x))

class Hidden_States():
    def __init__(self, num_features):
        self.weights = [0.5] * num_features
        self.bias = [0.5] * num_features
        self.features = []
        self.previous_feature = 0

    def forget_gate(self, feature):
        return sigmoid(sum(w * feature for w in self.weights) + self.bias[0])

    def input_gate(self, feature):
        gate_output = sigmoid(sum(w * feature for w in self.weights) + self.bias[0])
        model_feature = math.tanh(sum(w * self.previous_feature for w in self.weights) + self.bias[0])
        return gate_output, model_feature

    def output_gate(self, feature):
        gate_output = sigmoid(sum(w * feature for w in self.weights) + self.bias[0])
        return gate_output * self.previous_feature

    def backpropagation(self, output, feature, learning_rate):
        loss = cross_entropy(output, sum(self.weights))
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * (output / loss)
            self.bias[i] -= learning_rate * (output / loss)

class Sentiment_Analysis():
    def __init__(self, num_features):
        self.weights = [0.5] * num_features
        self.bias = [0.5] * num_features
        self.features = []
        self.pos = 0

    def forward(self, feature):
        if self.pos >= len(self.weights):
            return
        feature = feature * self.weights[self.pos] + self.bias[self.pos]
        self.features.append(feature)
        self.pos += 1
        self.forward(feature)

    def backpropagation(self, output: int, learning_rate: float):
        for i in range(len(self.weights)):
            delta_weights = self.weights[i] - output
            delta_bias = self.bias[i] - output
            gradient_weights = output / cross_entropy(output, delta_weights)
            gradient_bias = output / cross_entropy(output, delta_bias)
            self.weights[i] -= learning_rate * gradient_weights
            self.bias[i] -= learning_rate * gradient_bias

class ConvolutionNetwork():
    def __init__(self, image, stride, filter1):
        self.image = image
        self.stride = stride
        self.filter1 = filter1
        self.weights = [0.5] * len(image)
        self.bias = [0.5] * len(image)

    def forward(self, feature):
        for i in range(len(self.weights)):
            feature = feature * self.weights[i] + self.bias[i]
        return feature

    def backpropagation(self, output, learning_rate):
        for i in range(len(self.weights)):
            feature = self.weights[i] + self.bias[i]
            n_log = cross_entropy(feature, output)
            self.weights[i] -= learning_rate * n_log / output
            self.bias[i] -= learning_rate * n_log / output

    def apply_filter(self):
        for px in range(0, len(self.image), self.stride):
            self.image[px] *= self.filter1

    def max_pooling(self):
        return max(self.image[::self.stride])

def main():
    input_feature = 1.0
    expected_output = 0.7
    learning_rate = 0.01

    sentiment_analysis = Sentiment_Analysis(num_features=5)
    convolution_network = ConvolutionNetwork(image=[0.5, 0.3, 0.7], stride=1, filter1=0.9)

    print("Running Sentiment Analysis (RNN)...")
    sentiment_analysis.forward(input_feature)
    sentiment_analysis.backpropagation(expected_output, learning_rate)

    print("Running Convolution Network (CNN)...")
    convolution_network.forward(input_feature)
    convolution_network.backpropagation(expected_output, learning_rate)

    print("Applying filters and performing max pooling...")
    convolution_network.apply_filter()
    max_pooled_value = convolution_network.max_pooling()

    print("Max pooled value: ", max_pooled_value)
    print("Training complete.")

if __name__ == "__main__":
    main()
