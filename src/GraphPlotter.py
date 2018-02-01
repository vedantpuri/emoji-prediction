from matplotlib import pyplot as plt
from NaiveBayes import NaiveBayes


class GraphPlotter:
    @staticmethod
    def make_accuracy_graph(input_tweets_file, input_labels_file):

        """
            Creates a graph to determine the relationship between alpha and accuracy.
            The values of that are tried are from 1 - 50
            The test set size is kept as default - 20%
        """

        plt.title("Accuracy of Naive Bayes with varied alpha and fixed test set size")
        plt.xlabel("Alpha Value")
        plt.ylabel("Accuracy Achieved")

        accuracies = []

        for i in range(1, 51):
            nb = NaiveBayes(input_tweets_file, input_labels_file, i)
            nb.update_model()
            accuracies.append(nb.evaluate_classifier_accuracy())

        plt.plot(range(1, 51), accuracies)
        # Save the figure in the Figures directory
        file_name = "Figures/accuracy_varied_alpha_" + input_tweets_file[8:10] + ".jpeg"
        plt.savefig(file_name)
        # To refresh the graph
        plt.close()

    @staticmethod
    def make_data_percentage_graph(input_tweets_file, input_labels_file):

        """
            Creates a graph to determine the relationship between test set size and accuracy.
            We try testing on 25 percent - 85 percent of the dataset, where we increment by 5 percent in every run
            The value of alpha is kept as default - 1.0
        """
        plt.title("Accuracy of Naive Bayes with varied test data size and fixed alpha")
        plt.xlabel("Percentage of data used for testing")
        plt.ylabel("Accuracy Achieved")

        values = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0]
        accuracies = []

        for val in values:
            nb = NaiveBayes(input_tweets_file, input_labels_file, 1.0, (val / 100.0))
            nb.update_model()
            accuracies.append(nb.evaluate_classifier_accuracy())

        plt.plot(values, accuracies)
        # Save the figure in the Figures directory
        file_name = "Figures/accuracy_varied_test_ratio_" + input_tweets_file[8:10] + ".jpeg"
        plt.savefig(file_name)
        # To refresh the graph
        plt.close()

    @staticmethod
    def save_graph():
        # Save English graphs
        GraphPlotter.make_accuracy_graph("../data/us_trial.text", "../data/us_trial.labels")
        GraphPlotter.make_data_percentage_graph("../data/us_trial.text", "../data/us_trial.labels")
        print("english done")
        # Save spanish graphs
        print("spanish started")
        GraphPlotter.make_accuracy_graph("../data/es_trial.text", "../data/es_trial.labels")
        GraphPlotter.make_data_percentage_graph("../data/es_trial.text", "../data/es_trial.labels")
        print("spanish done")

print("started")
GraphPlotter.save_graph()
print("done")








