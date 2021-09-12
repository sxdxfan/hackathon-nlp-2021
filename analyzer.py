import re
import docx
import pickle
import nltk
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


class Analyzer:
    def __init__(self, binary_classifier_path, vectorizer_path, factor_classifier_path):
        nltk.download('stopwords')
        self.binary_classifier = CatBoostClassifier()
        self.binary_classifier.load_model(binary_classifier_path)
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        self.factor_classifier = pickle.load(open(factor_classifier_path, 'rb'))
        self.factor_label_map = [
            '3_1\nширота дискреционных полномочий',
            '3_2\nопределение компетенции по формуле "вправе"',
            '3_5\nпринятие нормативного правового акта за пределами компетенции',
            '3_9\nнормативные коллизии - противоречия, в том числе внутренние, между нормами',
            '4_1\nналичие завышенных требований к лицу, предъявляемых для реализации принадлежащего ему права',
            '4_3\nюридико-лингвистическая неопределенность',
            '3_3\nвыборочное изменение объема прав',
            '3_7\nотсутствие или неполнота административных процедур',
            '4_2\nзлоупотребление правом заявителя государственными органами, органами местного самоуправления или организациями (их должностными лицами)',
            '3_4\nчрезмерная свобода подзаконного нормотворчества',
            '3_6\nзаполнение законодательных пробелов при помощи подзаконных актов в отсутствие законодательной делегации соответствующих полномочий',
            '3_8\nотказ от конкурсных (аукционных) процедур'
        ]

    @staticmethod
    def get_text(document_path):
        doc = docx.Document(document_path)
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                full_text.append(paragraph.text)
        return full_text

    @staticmethod
    def preprocess_and_create_features(data):
        data['line'] = data['line'].apply(lambda x: re.sub(r'\s', ' ', str(x).strip()))
        data['len_line'] = data['line'].apply(lambda x: len(x))
        data['first_word'] = data['line'].apply(lambda x: x.split(' ')[0].strip().lower())
        data['last_word'] = data['line'].apply(lambda x: x.split(' ')[-1].strip().lower())
        data['count_decimal'] = data['line'].apply(lambda x: len(re.findall(r'[0-9]+', x)))
        data['count_uppercased_chars'] = data['line'].apply(lambda x: len(re.findall(r'[A-ZA-Я]+', x)))
        data['count_stopwords'] = data['line'].apply(
            lambda x: len([token for token in x.split(' ') if token in nltk.corpus.stopwords.words('russian')])
        )
        data['count_puncts'] = data['line'].apply(lambda x: len(re.findall(r'[.,:?;!)(*%#]\"\'', x)))

        return data

    def get_feature_importances(self, label):
        feature_importances = pd.DataFrame(
            np.c_[list(self.vectorizer.vocabulary_),
            self.factor_classifier.coef_[label]],
            columns=['word', 'coef']
        )
        feature_importances['coef'] = feature_importances['coef'].astype('float64')
        return feature_importances.sort_values(by='coef', ascending=False)

    @staticmethod
    def create_features_from_feature_importances(data, feature_importances):
        best_words = feature_importances['word'].head(10).values
        for word in best_words:
            col_name = 'gram_' + word.replace(' ', '_') + '_flg'
            data[col_name] = data['line'].apply(lambda x: 1 if word.lower() in x else 0)
        return data

    def create_additional_features(self, data):
        for i in range(len(self.factor_label_map)):
            feature_importances = self.get_feature_importances(i)
            data = self.create_features_from_feature_importances(data, feature_importances)
        return data

    def predict_factor(self, text):
        if not text.strip():
            return None

        vectorized_text = self.vectorizer.transform([text])
        predicted_factor = self.factor_classifier.predict(vectorized_text)[0]
        return self.factor_label_map[predicted_factor]

    def analyze(self, document_path):
        corrupt_texts = {}

        full_text = self.get_text(document_path)
        paragraphs_df = pd.DataFrame(full_text, columns=['line'])
        data = self.preprocess_and_create_features(paragraphs_df)
        data = self.create_additional_features(data)
        cols = list(data)
        y_proba = self.binary_classifier.predict_proba(data[cols])
        y_pred = np.where(y_proba[:, 1] > 0.5, 1, 0)

        last_normal_paragraph = ''
        current_corrupt_paragraph = ''

        for text, prediction in zip(data['line'], y_pred):
            if prediction:
                current_corrupt_paragraph += text + '\n'
            else:
                if current_corrupt_paragraph:
                    predicted_factor = self.predict_factor(current_corrupt_paragraph)
                    if predicted_factor not in corrupt_texts:
                        corrupt_texts[predicted_factor] = []
                    corrupt_texts[predicted_factor] += [[
                        last_normal_paragraph,
                        current_corrupt_paragraph,
                        text
                    ]]
                    current_corrupt_paragraph = ''
                last_normal_paragraph = text

        return corrupt_texts
