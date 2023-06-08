import string
from classifier import BayesClassifier

# Converts text into a list of words
def process_text(text):
    new_str = text.translate(str.maketrans('', '', string.punctuation))
    preprocessed_text = new_str.lower().split()

    return preprocessed_text

# Builds a list of words that are seen
def build_vocab(preprocessed_text):
    vocab = []

    for word in preprocessed_text:
        if word not in vocab and not any(char.isdigit() for char in word):
            vocab.append(word)

    vocab = sorted(vocab)

    return vocab

# Converts a line of text into a vector and label
def vectorize_text(text, vocab):
    vectorized_text = []
    for word in vocab:
        if word in text:
            vectorized_text.append('1')
        else: vectorized_text.append('0')

    label = text[-1]

    return vectorized_text, label

def readfile(file_name):
    with open(file_name, 'r') as file:
        return file.read()
    
def data_to_vectors(data, vocab):
    vectors = []
    labels = []

    for line in data.split('\n'):
        processed_line = process_text(line)

        if len(processed_line) > 0:
            (vector, label) = vectorize_text(processed_line, vocab)
            vectors.append(vector)
            labels.append(label)

    return (vectors, labels)

def createprocessedfile(vectors, labels, vocab, file_name):
    with open(file_name, 'w') as file:
        # Write top line
        for item in vocab:
            file.write("%s," % item)

        file.write('\n')

        # Write each vector to file
        for i, vector in enumerate(vectors):
            for item in vector:
                file.write("%s," % item)

            file.write("%s\n" % labels[i])

def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    accuracy_score = 0

    for (prediction, actual) in zip(predicted_labels, true_labels):
        if prediction == actual:
            accuracy_score += 1

    accuracy_score /= len(predicted_labels)
    return accuracy_score*100

def main():
    # Take in text files and outputs sentiment scores
    test_data = readfile('testSet.txt')
    training_data = readfile('trainingSet.txt')

    training_vocab = build_vocab(process_text(training_data))
    test_vocab = build_vocab(process_text(test_data))

    (training_vectors, training_labels) = data_to_vectors(training_data, training_vocab)
    (test_vectors, test_labels) = data_to_vectors(test_data, test_vocab)

    createprocessedfile(test_vectors, test_labels, test_vocab, 'preprocessed_test.txt')
    createprocessedfile(training_vectors, training_labels, training_vocab, 'preprocessed_train.txt')

    classifier = BayesClassifier()

    percent_data_trained = 0
    with open('results.txt', 'w') as results:
        results.write("Percent Data Trained;Training data classification accuracy;Test data classification accuracy\n")

        for i in range(4):
            percent_data_trained = round((classifier.file_sections[i] / classifier.file_length) * 100, 2)
            classifier.train(training_vectors, training_labels, training_vocab, i)

            # Test on training set
            predictions = classifier.classify_text(training_vectors, training_vocab)
            _accuracy = round(accuracy(predictions, training_labels), 2)

            # Test on test set
            predictions = classifier.classify_text(test_vectors, test_vocab)
            _test_accuracy = round(accuracy(predictions, test_labels), 2)

            results.write(f"{percent_data_trained};{_accuracy};{_test_accuracy}\n")

    return 1


if __name__ == "__main__":
    main()
