import math

class BayesClassifier():
    """
    Naive Bayes Classifier
    file length: file length of training file
    sections: sections for incremental training
    """
    def __init__(self):
        self.positive_word_counts = {}
        self.negative_word_counts = {}
        self.number_positive_sentences = 0
        self.number_negative_sentences = 0
        self.percent_positive_sentences = 0
        self.percent_negative_sentences = 0
        self.file_length = 499
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2]


    def train(self, train_vectors, train_labels, vocab):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """

        # Calculate percentage possibility of positive or negative
        for label in train_labels:
            if label == '1':
                self.number_positive_sentences += 1
            else:
                self.number_negative_sentences += 1

        self.percent_positive_sentences = (self.number_positive_sentences / len(train_labels))
        self.percent_negative_sentences = (self.number_negative_sentences / len(train_labels))

        # Calculate amount for each word
        for i, word in enumerate(vocab):
            self.positive_word_counts[word] = 1
            self.negative_word_counts[word] = 1

            for vector, label in zip(train_vectors, train_labels):
                if label == '1' and vector[i] == '1':
                    self.positive_word_counts[word] += 1
                elif vector[i] == '1':
                    self.negative_word_counts[word] += 1

        return 1


    def classify_text(self, vectors, vocab):
        """
        vectors: [vector1, vector2, ...]
        predictions: [0, 1, ...]
        """

        predictions = []

        for vector in vectors:
            percent_pos = math.log(self.percent_positive_sentences)
            percent_neg = math.log(self.percent_negative_sentences)

            for seen, word in zip(vector, vocab):
                if word in self.positive_word_counts and word in self.negative_word_counts:
                    if (seen == '1'):
                        percent_pos += math.log((self.positive_word_counts[word]) / (self.number_positive_sentences+len(vocab)))
                    else:
                        percent_neg += math.log((self.negative_word_counts[word]) / (self.number_negative_sentences+len(vocab)))

            if percent_pos > percent_neg:
                predictions.append('1')
            else:
                predictions.append('0')

        return predictions
    
