
"""
 Yusuf Alptug Aydin
 260201065

 600 epochs:
 Train accuracy: 89.163
 Test accuracy: 89.58
"""

import torch
import dlc_practical_prologue_edited as prologue
import UtilsProvider as Provider

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class NeuralNetwork_3L:

    def __init__(self, activation_functions, hidden_units_l1, hidden_units_l2, step_size, epsilon, factor,
                 train_input, train_target, test_input, test_target
                 ):

        self.utils = Provider.UtilsProvider()

        self.activation_functions = activation_functions
        self.hidden_units_l1 = hidden_units_l1
        self.hidden_units_l2 = hidden_units_l2
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
    # 1 hidden layer : activation_types[1]
    # 1 output layer : activation_types[2]
    def forward_pass(self, activation_types, w1, b1, w2, b2, w3, b3, x):
        s_1 = w1 @ x + b1  # dot product + bias : weighted sum
        x_1 = self.utils.activation(activation_types[0], s_1)

        s_2 = w2 @ x_1 + b2  # dot product + bias : weighted sum
        x_2 = self.utils.activation(activation_types[1], s_2)

        s_3 = w3 @ x_2 + b3  # dot product + bias : weighted sum
        x_3 = self.utils.activation(activation_types[2], s_3)

        return x, s_1, x_1, s_2, x_2, s_3, x_3

    # output layer : activation_types[2]
    # hidden layer : activation_types[1]
    # hidden layer : activation_types[0]
    def backward_pass(self, activation_types,
                      w1, b1, w2, b2, w3, b3,
                      t,
                      x, s1, x1, s2, x2, s3, x3,
                      dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3):

        delta4 = self.utils.der_mse(x3, t)  # x3 - t # [10]

        term3 = (delta4 * self.utils.der_activation(activation_types[2], s3))  # [10]

        delta3 = w3.t().mm(term3.view(-1, 1))  # [100,1] = [100,10] . [10,1]
        term2 = (delta3.squeeze() * self.utils.der_activation(activation_types[1],
                                                              s2))  # [100,1] = [100,1] .* [100] -> [100] .* [100]

        delta2 = w2.t().mm(term2.view(-1, 1))
        term1 = (delta2.squeeze() * self.utils.der_activation(activation_types[0], s1)).view(-1, 1)

        dl_dw3.add_(term3.view(-1, 1).mm(x2.t().view(1, -1)))  # [10,100] = [10, 1] . [1,100]
        dl_db3.add_(term3)  # [10] = [10] .* [10]

        dl_dw2.add_(term2.view(-1, 1).mm(x1.t().view(1, -1)))  # [100,300] = [100, 1] . [1,300]
        dl_db2.add_(term2)  # [100] = [100] .* [100]

        dl_dw1.add_((term1).view(-1, 1).mm(x.t().view(1, -1)))  # [300,784] = [300,1] . [1,784]
        dl_db1.add_(term1.squeeze())  # [300] = [300,1] --> [300]

        return dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3  # [300, 784], [300], [100, 300], [100], [10, 100], [10]

    def init_weights_and_biases(self):
        self.w1 = torch.empty(self.hidden_units_l1, self.features_count).normal_(0, self.epsilon)  # size = 300 * 784
        self.b1 = torch.empty(self.hidden_units_l1).normal_(0, self.epsilon)  # size = 300

        self.w2 = torch.empty(self.hidden_units_l2, self.hidden_units_l1).normal_(0, self.epsilon)  # size = 100 * 300
        self.b2 = torch.empty(self.hidden_units_l2).normal_(0, self.epsilon)  # size = 100

        self.w3 = torch.empty(self.classes_count, self.hidden_units_l2).normal_(0, self.epsilon)  # size = 10 * 100
        self.b3 = torch.empty(self.classes_count).normal_(0, self.epsilon)  # size = 10

    def update_weights_and_biases(self, dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3):
        self.w1 -= self.step_size * dl_dw1
        self.w2 -= self.step_size * dl_dw2
        self.w3 -= self.step_size * dl_dw3

        self.b1 -= self.step_size * dl_db1
        self.b2 -= self.step_size * dl_db2
        self.b3 -= self.step_size * dl_db3

    def set_weights_and_biases(self, w1, b1, w2, b2, w3, b3):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def train(self, epochs):
        import time
        start = time.perf_counter()

        # [forward pass -> calculate error -> backward pass] -> update weights
        for i in range(epochs):
            # Reset variables before each epoch
            cum_loss = 0
            train_accuracy = 0
            test_accuracy = 0
            dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3 = self.reset_terms()

            for j in range(self.train_input_count):  # forward -> calculate error -> backward
                x0, s1, x1, s2, x2, s3, x3 = self.forward_pass(self.activation_functions,
                                                               self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                                                               self.train_input[j]
                                                               )

                train_accuracy += int(self.train_target[j, x3.argmax(0)] != 0)  # x3 -> predicted
                cum_loss += self.utils.mean_squared_error(x3, self.train_target[j])  # summing loss cumulatively

                dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3 = self.backward_pass(
                    self.activation_functions,
                    self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                    self.train_target[j],
                    x0, s1, x1, s2, x2, s3, x3,
                    dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3
                )

            self.update_weights_and_biases(dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3)  # updating part

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
        dl_dw3 = torch.zeros(self.w3.size())

        dl_db1 = torch.zeros(self.b1.size())
        dl_db2 = torch.zeros(self.b2.size())
        dl_db3 = torch.zeros(self.b3.size())
        return dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3

    def test(self):
        test_accuracy = 0
        for i in range(self.test_input_count):
            x0, s1, x1, s2, x2, s3, x3 = self.forward_pass(
                                                            self.activation_functions,
                                                            self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
                                                            self.test_input[i]
                                                            )
            test_accuracy += int(self.test_target[i, x3.argmax(0)] != 0)  # shape of test_target: 10000 x 10, x3.argmax(0) gives the index of maximum value
        return test_accuracy


def init_hyperparameters(train_input_size):
    # HYPERPARAMETERS
    activation_functions = ["sigmoid", "tanh", "sigmoid"]
    hidden_units_l1 = 300
    hidden_units_l2 = 100
    step_size = 0.1 / train_input_size
    epsilon = 0.1
    factor = 0.9
    epochs = 600  # epoch: one forward and one backward pass of entire training set

    return activation_functions, factor, hidden_units_l1, hidden_units_l2, step_size, epsilon, epochs


def init_hyperparameters_and_get_the_model(pretrained=False, save=False):
    # DATA LOADING
    train_input, train_target, test_input, test_target = prologue.load_data(one_hot_labels=True,
                                                                            normalize=True,
                                                                            full=True)
    # SETTING THINGS UP
    activation_functions, factor, hidden_units_l1, hidden_units_l2, step_size, epsilon, epochs = init_hyperparameters(
        train_input.size(0))

    train_input = train_input * factor
    test_input = test_input * factor

    # CREATING NETWORK
    network = NeuralNetwork_3L(activation_functions, hidden_units_l1, hidden_units_l2, step_size, epsilon, factor,
                               train_input, train_target, test_input, test_target)

    if (pretrained == True):
        network = load_and_set_weights(network)

    else:
        network.train(epochs)

        if (save):
            save_weights(network)

    return network


# SAVING TRAINED WEIGHTS
def save_weights(network):
    pretrained_weights = {
        'w1': network.w1,
        "w2": network.w2,
        "w3": network.w3,

        "b1": network.b1,
        "b2": network.b2,
        "b3": network.b3
    }
    torch.save(pretrained_weights, "pretrained_weights_QB.pt")


# LOAD PRE-TRAINED WEIGHTS AND TEST WITH TEST DATA
def load_and_set_weights(network):
    loaded_pretrained_weights = torch.load("pretrained_weights_QB.pt")

    l_w1 = loaded_pretrained_weights["w1"]
    l_w2 = loaded_pretrained_weights["w2"]
    l_w3 = loaded_pretrained_weights["w3"]

    l_b1 = loaded_pretrained_weights["b1"]
    l_b2 = loaded_pretrained_weights["b2"]
    l_b3 = loaded_pretrained_weights["b3"]

    network.set_weights_and_biases(l_w1.data, l_b1, l_w2.data, l_b2, l_w3.data, l_b3)
    return network


def get_test_accuracy_of_network(network):
    test_acc = network.test()
    result = test_acc * 100 / network.test_input_count
    return result


def run():
    network = init_hyperparameters_and_get_the_model(pretrained=True)  # (pretrained=False, save=True)
    test_acc = get_test_accuracy_of_network(network)
    print("Test accuracy for architecture 2 is: ", test_acc)
