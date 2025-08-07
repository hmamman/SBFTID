from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=100),
    'svm': SVC(probability=True, kernel="rbf", gamma="scale"),
}


def dnn_classifier(input_shape=(None, 13), nb_classes=2):
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Input, Dense

    model = Sequential([
        Input(shape=input_shape[1:]),
        Dense(30, activation="relu"),
        Dense(20, activation="relu"),
        Dense(15, activation="relu"),
        Dense(10, activation="relu"),
        Dense(5, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer="nadam", metrics=["accuracy"])

    return model