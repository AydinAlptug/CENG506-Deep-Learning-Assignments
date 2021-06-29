
"""
 Yusuf Alptug Aydin
 260201065

 1000 epochs:
 Train accuracy: 85.39
 Test accuracy: 86.53
"""

import torch
import dlc_practical_prologue_edited as prologue
import UtilsProvider as Provider

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class NeuralNetwork_2L:

    def __init__(self, activation_functions, hidden_units, step_size, epsilon, factor,
                 train_input, train_target, test_input, test_target
                 ):

        self.utils = Provider.UtilsProvider()

        self.activation_functions = activation_functions
        self.hidden_units = hidden_units
        self.step_size = step_size
        self.epsilon = epsilon
        self.factor = factor

        self.train_input = train_input * factor
        self.train_target = train_target
        self.test_input = test_input * factor
        self.test_target = test_target

        self.features_count = train_input.size(1)  # 784
        self.classes_count = train_target.size(1)  # 10
        self.train_input_count = train_input.size(0)
        self.test_input_count = test_input.size(0)

        self.init_weights_and_biases()

    # 1 hidden layer : activation_types[0]
    # 1 output layer : activation_types[1]

    def forward_pass(self, activation_types, w1, b1, w2, b2, x):
        s_1 = w1 @ x + b1  # dot product + bias : weighted sum
        x_1 = self.utils.activation(activation_types[0], s_1)

        s_2 = w2 @ x_1 + b2  # dot product + bias : weighted sum
        x_2 = self.utils.activation(activation_types[1], s_2)

        return x, s_1, x_1, s_2, x_2

    # output layer : activation_types[1]
    # hidden layer : activation_types[0]
    def backward_pass(self, activation_types,
                      w1, b1, w2, b2,
                      t,
                      x, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2):

        delta3 = self.utils.der_mse(x2, t)  # x2 - t # [10]
        term2 = (delta3 * self.utils.der_activation(activation_types[1], s2))  # [10]
        delta2 = w2.t().mm(term2.view(-1, 1))  # [300,1] = [300,10] . [10,1]

        dl_dw2.add_(
            term2.view(-1, 1).mm(x1.view(-1, 1).t().view(1, -1)))  # [10,300] = [10, 1] . [1,300]
        dl_db2.add_(term2)  # [10] = [10] .* [10]

        term1 = (delta2.squeeze() * self.utils.der_activation(activation_types[1], s1)).view(-1,
                                                                                             1)  # [300,1] = [300,1] .* [300] -> [300] .* [300]
        dl_dw1.add_(
            (term1).view(-1, 1).mm(x.view(-1, 1).t().view(1, -1)))  # [300,784] = [300,1] . [1,784]
        dl_db1.add_(term1.squeeze())  # [300] = [300,1] --> [300]

        return dl_dw1, dl_db1, dl_dw2, dl_db2  # [300,784], [300], [10,300], [10]

    def init_weights_and_biases(self):
        self.w1 = torch.empty(self.hidden_units, self.features_count).normal_(0, self.epsilon)  # size = 300 * 784
        self.b1 = torch.empty(self.hidden_units).normal_(0, self.epsilon)  # size = 300

        self.w2 = torch.empty(self.classes_count, self.hidden_units).normal_(0, self.epsilon)  # size = 10 * 300
        self.b2 = torch.empty(self.classes_count).normal_(0, self.epsilon)  # size = 10

    def update_weights_and_biases(self, dl_dw1, dl_db1, dl_dw2, dl_db2):
        self.w1 -= self.step_size * dl_dw1
        self.w2 -= self.step_size * dl_dw2

        self.b1 -= self.step_size * dl_db1
        self.b2 -= self.step_size * dl_db2

    def set_weights_and_biases(self, w1, b1, w2, b2):
        self.w1 = w1
        self.w2 = w2

        self.b1 = b1
        self.b2 = b2

    def train(self, epochs):
        import time
        start = time.perf_counter()

        # [forward pass -> calculate error -> backward pass] -> update weights
        for i in range(epochs):
            # Reset variables before each epoch
            cum_loss = 0
            train_accuracy = 0
            test_accuracy = 0
            dl_dw1, dl_db1, dl_dw2, dl_db2 = self.reset_terms()

            for j in range(self.train_input_count):  # forward -> calculate error -> backward
                x0, s1, x1, s2, x2 = self.forward_pass(self.activation_functions,
                                                       self.w1, self.b1, self.w2, self.b2,
                                                       self.train_input[j]
                                                       )

                train_accuracy += int(self.train_target[j, x2.argmax(0)] != 0)  # x2 -> predicted
                cum_loss += self.utils.mean_squared_error(x2, self.train_target[j])  # summing loss cumulatively

                dl_dw1, dl_db1, dl_dw2, dl_db2 = self.backward_pass(
                    self.activation_functions,
                    self.w1, self.b1, self.w2, self.b2,
                    self.train_target[j],
                    x0, s1, x1, s2, x2,
                    dl_dw1, dl_db1, dl_dw2, dl_db2
                )

            self.update_weights_and_biases(dl_dw1, dl_db1, dl_dw2, dl_db2)  # updating part

            # Calculating test accuracy on each epochs
            test_accuracy += self.test()

            train_accuracy = (train_accuracy) * 100 / self.train_input_count
            test_accuracy = (test_accuracy) * 100 / self.test_input_count

            print("Epoch #", i, " || Cumulative Loss:", cum_loss.item(), "|| Train Accuracy:", train_accuracy,
                  "|| Test Accuracy:", test_accuracy)
            print("Execution time: ", ((time.perf_counter() - start) / 60), "minutes")

    def reset_terms(self):
        dl_dw1 = torch.zeros(self.w1.size())
        dl_dw2 = torch.zeros(self.w2.size())

        dl_db1 = torch.zeros(self.b1.size())
        dl_db2 = torch.zeros(self.b2.size())

        return dl_dw1, dl_db1, dl_dw2, dl_db2

    def test(self):
        test_accuracy = 0
        for i in range(self.test_input_count):
            x0, s1, x1, s2, x2 = self.forward_pass(
                self.activation_functions,
                self.w1, self.b1, self.w2, self.b2,
                self.test_input[i])
            test_accuracy += int(self.test_target[i, x2.argmax(0)] != 0)
        return test_accuracy


def init_hyperparameters(train_input_size):
    activation_functions = ["tanh", "tanh"]
    factor = 0.9
    hidden_units = 300
    step_size = 0.01 / train_input_size
    epsilon = 1e-3
    epochs = 1000  # epoch: one forward and one backward pass of entire training set

    return activation_functions, factor, hidden_units, step_size, epsilon, epochs


def init_hyperparameters_and_get_the_model(pretrained=False, save=False):
    # DATA LOADING
    train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True,
                                                                            normalize=True,
                                                                            full=True)
    # SETTING THINGS UP
    activation_functions, factor, hidden_units, step_size, epsilon, epochs = init_hyperparameters(train_input.size(0))

    train_input = train_input * factor
    test_input = test_input * factor

    # CREATING NETWORK
    network = NeuralNetwork_2L(activation_functions, hidden_units, step_size, epsilon, factor,
                               train_input, train_target, test_input, test_target)

    if (pretrained == True):
        network = load_and_set_weights(network)

    else:
        network.train(epochs)

        if (save):
            save_weights(network)

    return network


def save_weights(network):
    pretrained_weights = {

        'w1': network.w1,
        "w2": network.w2,

        "b1": network.b1,
        "b2": network.b2
    }
    torch.save(pretrained_weights, "pretrained_weights_QA.pt")


# LOAD PRE-TRAINED WEIGHTS AND TEST WITH TEST DATA
def load_and_set_weights(network):
    loaded_pretrained_weights = torch.load("pretrained_weights_QA.pt")

    l_w1 = loaded_pretrained_weights["w1"]
    l_w2 = loaded_pretrained_weights["w2"]

    l_b1 = loaded_pretrained_weights["b1"]
    l_b2 = loaded_pretrained_weights["b2"]

    network.set_weights_and_biases(l_w1.data, l_b1, l_w2.data, l_b2)

    return network


def get_test_accuracy_of_network(network):
    test_acc = network.test()
    result = test_acc * 100 / network.test_input_count
    return result


def run():
    network_1 = init_hyperparameters_and_get_the_model(pretrained=True)  # (pretrained=False, save=True)
    test_acc_1 = get_test_accuracy_of_network(network_1)
    print("Test accuracy for architecture 1 is: ", test_acc_1)
