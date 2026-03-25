# 1. Import bibliotek
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import numpy as np

# 2. Wczytanie danych
iris = load_iris()
X = iris.data #X – macierz cech (4 kolumny: sepal length, sepal width, petal length, petal width).
y = iris.target #y – wektor etykiet (0, 1, 2 odpowiadające trzem gatunkom irysów: setosa, versicolor, virginica).


feature_names = iris.feature_names
target_names = iris.target_names

print("Cechy:", feature_names)
print("Klasy:", target_names)

# 3. Podział danych na treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) #test_size=0.3 – 30% danych idzie na test, 70% na trening. ustala „ziarno losowości”. Dzięki temu podział będzie zawsze taki sam, jeśli uruchomisz kod kilka razy. Bez tego wyniki mogłyby się różnić przy każdym uruchomieniu.

# 4. Tworzenie modelu Random Forest
model = RandomForestClassifier(
    n_estimators=200,   # liczba drzew
    random_state=42     #wytlumacz co to jest
)

# 5. Trenowanie modelu
model.fit(X_train, y_train)

# 6. Predykcja
y_pred = model.predict(X_test)

# 7. Ocena modelu
print("\nDokładność:", accuracy_score(y_test, y_pred))

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\nMacierz pomyłek:")
print(confusion_matrix(y_test, y_pred))

# 8. Ważność cech
print("\nWażność cech:")
for name, importance in zip(feature_names, model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# 9. Przykładowa predykcja dla nowych danych
# (np. jeden kwiat)
samples = [
    [5.1, 3.5, 1.4, 0.2],  # setosa
    [6.0, 2.9, 4.5, 1.5],  # versicolor
    [6.5, 3.0, 5.5, 2.0],  # virginica
    [5.5, 2.5, 4.0, 1.3],  # versicolor
    [7.2, 3.2, 6.0, 2.5]   # virginica
]

for sample in samples:
    prediction = model.predict([sample])
    print(f"\nDane: {sample}")
    print("Klasa:", target_names[prediction][0])

print("\nPredykcja dla próbki:", sample)
print("Przewidziana klasa:", target_names[prediction][0])



import matplotlib.pyplot as plt

# Wybieramy 2 najważniejsze cechy (petal length i petal width)
X_vis = X[:, 2:4]  # kolumny: petal length, petal width

plt.figure()
for i, target_name in enumerate(target_names):
    plt.scatter(
        X_vis[y == i, 0],  # petal length
        X_vis[y == i, 1],  # petal width
        label=target_name
    )

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Wizualizacja danych Iris")
plt.legend()

plt.show()


# Wybieramy pierwsze drzewo z lasu
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