import sys
import file_paths

def run_nb():
    print("Importing Naive Bayes...")
    from NaiveBayes import NaiveBayes
    print("Successfully Imported Naive Bayes.")
    print("Running...")
    nb_obj = NaiveBayes(file_paths.us_tweets_path, file_paths.us_labels_path)
    nb_obj.update_model()
    print("Execution Complete. Accuracy:" + str(nb_obj.evaluate_classifier_accuracy()) + " %")

def run_dtc():
    print("Importing Decision Tree Classifier...")
    # Import dtc here
    print("Successfully Imported Decision Tree Classifier.")
    print("Running...")
    # Commands to run
    # print("Execution Complete. Accuracy:" + *insert eval function here* + " %")

def run_lstm():
    print("Importing LSTM...")
    # Import lstm here
    print("Successfully Imported LSTM.")
    print("Running...")
    # Commands to run
    # print("Execution Complete. Accuracy:" + *insert eval function here* + " %")

def run_blstm():
    print("Importing Bi-Directional LSTM...")
    # Import lstm here
    print("Successfully Imported Bi-Directional LSTM.")
    print("Running...")
    # Commands to run
    # print("Execution Complete. Accuracy:" + *insert eval function here* + " %")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Invalid Argument. Quitting...")
        exit()
    if sys.argv[1] in {"nb", "dtc", "lstm", "blstm"}:
        function_to_run = "run_" + sys.argv[1]
        globals()[function_to_run]()
    else:
        print("Invalid Argument. Quitting...")
        exit()
