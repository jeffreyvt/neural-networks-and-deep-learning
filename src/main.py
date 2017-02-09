import mnist_loader as ml
import network2 as network


if __name__ == "__main__":
    training_data, validation_data, test_data = ml.load_data_wrapper()


    net = network.Network([784, 30, 10], cost=network.CrossEntropyCost)
    net.SGD(training_data, 30, 10, 0.5,
            lmbda=5.0,
            evaluation_data=validation_data,
            monitor_evaluation_accuracy=True,
            monitor_evaluation_cost=True,
            monitor_training_accuracy=True,
            monitor_training_cost=True)