# =========================================
# 1. Importy
# =========================================
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# ================================
# 2. Wczytanie MNIST
# ================================
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)

X = mnist.data.to_numpy()   # <-- tu zamieniamy na numpy array
y = mnist.target.astype(int).to_numpy()  # <-- też na array

# =========================================
# 3. Podział danych
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 4. SKLEARN RANDOM FOREST
# =========================================
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== SKLEARN ===")
print("Accuracy:", accuracy_score(y_test, y_pred))


# =========================================
# 5. DRZEWO DECYZYJNE (Twoje)
# =========================================
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class MyDecisionTree:
    def __init__(self, max_depth=5):
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

        # 🔥 LOSOWANIE CECH (KLUCZOWE!)
        features = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)

        best_feature, best_thresh = self.best_split(X, y, features)

        if best_feature is None:
            return Node(value=self.most_common(y))

        left_idx = X[:, best_feature] <= best_thresh
        right_idx = X[:, best_feature] > best_thresh

        left = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right = self.build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def best_split(self, X, y, features):
        best_gini = 1
        split_idx, split_thresh = None, None

        for feature in features:
            thresholds = np.random.choice(X[:, feature], size=10)

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


# =========================================
# 6. RANDOM FOREST (Twój)
# =========================================
class MyRandomForest:
    def __init__(self, n_estimators=10, max_depth=5):
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


# =========================================
# 7. Trenowanie Twojego modelu
# =========================================
my_model = MyRandomForest(n_estimators=20, max_depth=5)
my_model.fit(X_train, y_train)

y_pred_my = my_model.predict(X_test)

print("\n=== MY MODEL ===")
print("Accuracy:", accuracy_score(y_test, y_pred_my))


# =========================================
# 8. Porównanie
# =========================================
print("\n=== PORÓWNANIE ===")
print("Sklearn RF:", accuracy_score(y_test, y_pred))
print("My RF     :", accuracy_score(y_test, y_pred_my))

# =========================================
# 9. Wizualizacja przykładowych cyfr
# =========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))

# pokażemy 10 pierwszych cyfr
for i in range(10):
    plt.subplot(2, 5, i+1)
    img = X[i].reshape(28, 28)  # 784 -> 28x28
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y[i]}")
    plt.axis('off')

plt.suptitle("Przykładowe cyfry z MNIST")
plt.show()