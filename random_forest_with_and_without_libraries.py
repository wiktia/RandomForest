# ================================
# 1. Import bibliotek
# ================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

# ================================
# 2. Wczytanie danych
# ================================
iris = load_iris()
X = iris.data #X – macierz cech (4 kolumny: sepal length, sepal width, petal length, petal width).
y = iris.target #y – wektor etykiet (0, 1, 2 odpowiadające trzem gatunkom irysów: setosa, versicolor, virginica).

feature_names = iris.feature_names
target_names = iris.target_names

print("Cechy:", feature_names)
print("Klasy:", target_names)

# ================================
# 3. Podział danych na treningowe i testowe
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# test_size=0.3 – 30% danych idzie na test, 70% na trening.
# random_state ustala „ziarno losowości”. Dzięki temu podział będzie zawsze taki sam.

# ================================
# 4. Random Forest (SKLEARN)
# ================================
model = RandomForestClassifier(
    n_estimators=200,   # liczba drzew
    random_state=42     # ziarno losowości (powtarzalność wyników)
)

# ================================
# 5. Trenowanie modelu
# ================================
model.fit(X_train, y_train)

# ================================
# 6. Predykcja
# ================================
y_pred = model.predict(X_test)

# ================================
# 7. Ocena modelu
# ================================
print("\n=== SKLEARN ===")
print("Dokładność:", accuracy_score(y_test, y_pred))

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))


# =====================================================
# 8. DRZEWO DECYZYJNE OD ZERA (BEZ SKLEARN)
# =====================================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class MyDecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # warunek stopu
        if depth >= self.max_depth or n_classes == 1:
            return Node(value=self.most_common(y))

        best_feature, best_thresh = self.best_split(X, y)

        if best_feature is None:
            return Node(value=self.most_common(y))

        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def best_split(self, X, y):
        best_gini = 1
        split_idx, split_thresh = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] <= t]
                right = y[X[:, feature] > t]

                if len(left) == 0 or len(right) == 0:
                    continue

                gini = self.gini(left, right)

                if gini < best_gini:
                    best_gini = gini
                    split_idx = feature
                    split_thresh = t

        return split_idx, split_thresh

    def gini(self, left, right):
        def gini_calc(group):
            classes = np.unique(group)
            score = 0
            for c in classes:
                p = np.sum(group == c) / len(group)
                score += p ** 2
            return 1 - score

        n = len(left) + len(right)
        return (len(left)/n)*gini_calc(left) + (len(right)/n)*gini_calc(right)

    def most_common(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)


# =====================================================
# 9. RANDOM FOREST OD ZERA
# =====================================================
class MyRandomForest:
    def __init__(self, n_estimators=10, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def bootstrap(self, X, y):
        n = len(X)
        idx = np.random.choice(n, n, replace=True)
        return X[idx], y[idx]

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            X_s, y_s = self.bootstrap(X, y)
            tree = MyDecisionTree(max_depth=self.max_depth)
            tree.fit(X_s, y_s)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])

        final = []
        for i in range(X.shape[0]):
            votes = preds[:, i]
            final.append(np.bincount(votes).argmax())

        return np.array(final)


# ================================
# 10. Trenowanie własnego modelu
# ================================
my_model = MyRandomForest(n_estimators=50, max_depth=10)
my_model.fit(X_train, y_train)

y_pred_my = my_model.predict(X_test)

print("\n=== MÓJ MODEL (BEZ SKLEARN) ===")
print("Dokładność:", accuracy_score(y_test, y_pred_my))
print(confusion_matrix(y_test, y_pred_my))


# ================================
# 11. PORÓWNANIE
# ================================
print("\n=== PORÓWNANIE ===")
print("Sklearn RF:", accuracy_score(y_test, y_pred))
print("My RF     :", accuracy_score(y_test, y_pred_my))


# ================================
# 12. Wizualizacja danych
# ================================
X_vis = X[:, 2:4]

plt.figure()
for i, target_name in enumerate(target_names):
    plt.scatter(
        X_vis[y == i, 0],
        X_vis[y == i, 1],
        label=target_name
    )

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Wizualizacja danych Iris")
plt.legend()
plt.show()


# ================================
# 13. Wizualizacja drzewa (tylko sklearn)
# ================================
estimator = model.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(estimator,
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          rounded=True,
          fontsize=10)

plt.title("Pierwsze drzewo z Random Forest")
plt.show()