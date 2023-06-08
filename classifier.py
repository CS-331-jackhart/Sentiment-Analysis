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
        self.file_sections = [self.file_length // 4, self.file_length // 3, self.file_length // 2, self.file_length]


    def train(self, train_vectors, train_labels, vocab, stage):
        """
        This function builds the word counts and sentence percentages used for classify_text
        train_data: vectorized text
        train_labels: vectorized labels
        vocab: vocab from build_vocab
        """

        # The amount of items we will look at in this training session
        end_idx = self.file_sections[stage]

        vectors = train_vectors[:end_idx] if stage <= 0 else train_vectors[self.file_sections[stage-1]:end_idx]
        labels = train_labels[:end_idx] if stage <= 0 else train_labels[self.file_sections[stage-1]:end_idx]

        # Total number of elements viewed
        train_length = end_idx

        # Calculate percentage possibility of positive or negative
        for label in labels:
            if label == '1':
                self.number_positive_sentences += 1
            else:
                self.number_negative_sentences += 1

        self.percent_positive_sentences = (self.number_positive_sentences / train_length)
        self.percent_negative_sentences = (self.number_negative_sentences / train_length)

        # Calculate amount for each word
        for i, word in enumerate(vocab):
            if word not in self.positive_word_counts: self.positive_word_counts[word] = 1
            if word not in self.negative_word_counts: self.negative_word_counts[word] = 1

            for vector, label in zip(vectors, labels):
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
            percent_pos = math.log(max(self.percent_positive_sentences, 1))
            percent_neg = math.log(max(self.percent_negative_sentences, 1))

            for seen, word in zip(vector, vocab):
                if word in self.positive_word_counts and word in self.negative_word_counts:
                    if (seen == '1'):
                        percent_pos += math.log((self.positive_word_counts[word]+1) / (self.number_positive_sentences+2))
                        percent_neg += math.log((self.negative_word_counts[word]+1) / (self.number_negative_sentences+2))

            if percent_pos > percent_neg:
                predictions.append('1')
            else:
                predictions.append('0')

        return predictions
    
