import string

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
    vocab.append('classlabel')

    return vocab

# Converts a line of text into a vector and label
def vectorize_text(text, vocab):
    vectorized_text = []
    for word in vocab:
        if word in text:
            vectorized_text.append(1)
        else: vectorized_text.append(0)

    label = text[-1]

    return vectorized_text, label


def accuracy(predicted_labels, true_labels):
    """
    predicted_labels: list of 0/1s predicted by classifier
    true_labels: list of 0/1s from text file
    return the accuracy of the predictions
    """

    return accuracy_score

def readfile(file_name):
    with open(file_name, 'r') as file:
        return file.read()

def createprocessedfile(data, file_name):
    processed_data = process_text(data)
    vocab = build_vocab(processed_data)

    with open(file_name, 'w') as file:
        # Write top line
        for item in vocab:
            file.write("%s," % item)

        file.write('\n')

        # Write each vector to file
        for line in data.split('\n'):
            processed_line = process_text(line)

            if len(processed_line) > 0:
                (vector, label) = vectorize_text(processed_line, vocab)
            
                for item in vector:
                    file.write("%s," % item)

            file.write("%s\n" % label)

def main():
    # Take in text files and outputs sentiment scores
    test_data = readfile('testSet.txt')
    training_data = readfile('trainingSet.txt')

    createprocessedfile(test_data, 'preprocessed_train.txt')
    createprocessedfile(training_data, 'preprocessed_test.txt')

    return 1


if __name__ == "__main__":
    main()
