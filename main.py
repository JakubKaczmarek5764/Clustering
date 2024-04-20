import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import StandardScaler
import functions
import ml_functions
species_names_dict = {
    0: 'setosa',
    1: 'versicolor',
    2: 'virginica'
}
names_in_polish_dict = {
    'sepal length': 'Długość działki kielicha (cm)',
    'sepal width': 'Szerokość działki kielicha (cm)',
    'petal length': 'Długość płatka (cm)',
    'petal width': 'Szerokość płatka (cm)'
}
# wczytywanie danych
df_cluster = pd.read_csv(r'data.csv', header=None) # wczytywanie danych do dataframe, r to raw string, czyli slashe sa interpretowane jako slashe, header=None sprawia, ze pierwszy wiersz jest interpretowany jako dane a nie jako nazwy kolumn
df_cluster.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species'] # nadanie nazw kolumnom w dataframe
df_train = pd.read_csv(r'data_train.csv', header=None)
df_train.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']
df_test = pd.read_csv(r'data_test.csv', header=None)
df_test.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'species']

# dane do k srednich
df_cluster = df_cluster.drop(columns='species', axis=1) # usuniecie kolumny species w danych do klasteryzacji, axis=1 okresla, ze chodzi nam o kolumny

# dane do knn

scaler = StandardScaler()

X_train = df_train[['sepal length', 'sepal width', 'petal length', 'petal width']]
X_train = scaler.fit_transform(X_train) # normalizacja danych treningowych, fit_transform tylko na danych treningowych, fit liczy srednia i odchylenie standardowe, transform odejmuje srednia od kazdej wartosci i dzieli przez odchylenie standardowe
X_train = pd.DataFrame(X_train, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
Y_train = df_train['species']

X_test = df_test[['sepal length', 'sepal width', 'petal length', 'petal width']]
X_test = scaler.transform(X_test) # normalizacja danych testowych
X_test = pd.DataFrame(X_test, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
Y_test = df_test['species']


# k srednich

my_model = ml_functions.MyKMeans(df_cluster, 3) #utworzenie obiektu my_model klasy MyKMeans z danymi df_cluster i liczba klastrow rowna 3
my_model.run() # wywolanie funkcji run
df_centroids = pd.DataFrame(my_model.centroids) # utworzenie dataframe z centroidami liczonymi przez my_model
df_centroids.columns = ['sepal length', 'sepal width', 'petal length', 'petal width']
plt.rcParams['font.size'] = 12 #ustawienie rozmiaru czcionki na wykresie


for x_label, y_label in combinations(df_train.columns[:-1], 2): #combinations zwraca nam dwuelementowe kombinacje elementow z listy columns, r - liczba elementow w kombinacji
    plt.scatter(data=df_cluster, x=x_label, y=y_label, c=my_model.labels, cmap='Spectral', edgecolors='k') # utworzenie wykresu punktowego z danymi df_cluster, c - podanie wartosci ktore beda mapowane na kolor podany w cmap, cmap - paleta kolorow, edgecolors - kolor krawedzi punktow na wykresie
    plt.scatter(data=df_centroids, x=x_label, y=y_label, c=df_centroids.index, cmap='Spectral', marker='D', edgecolors='aqua', s=100, linewidth=2) # marker - ksztalt punktow, s - rozmiar punktow, linewidth - szerokosc krawedzi, df_centroids.index bo w c podana musi byc sekwencja numerow
    plt.xlim(functions.minimum(df_cluster, x_label) - 0.3, functions.maximum(df_cluster, x_label) + 0.3) # ustawienie rozmiaru osi x
    plt.xlabel(names_in_polish_dict[x_label], fontsize=14) # ustawienie nazwy osi x
    plt.ylabel(names_in_polish_dict[y_label], fontsize=14) # ustawienie nazwy osi y
    plt.show() # narysowanie wykresu
my_params = [] # utworzenie listy przechowujacej tuple z parametrami do wykresow
for number_of_clusters in range(2,11):
    my_model = ml_functions.MyKMeans(df_cluster, number_of_clusters)
    my_model.run()
    my_params.append((number_of_clusters, my_model.wcss, my_model.num_of_iterations)) # dodanie tupla z parametrami do listy

my_params = pd.DataFrame(my_params, columns=['k', 'wcss', 'iterations']) # przeksztalcenie listy w dataframe, zeby wygodniej bylo utworzyc wykresy
plt.scatter(data=my_params, x='k', y='wcss') # utworzenie wykresu punktowego zaleznosci wcss od k
plt.plot('k', 'wcss', data=my_params) # nalozenie na poprzedni wykres linii
# arange tworzy nam liste z wartosciami oddalonymi od siebie o step, lewa granica jest domknieta, prawa otwarta
plt.yticks(np.arange(functions.floor_to_ten(functions.minimum(my_params, 'wcss')), functions.ceil_to_ten(functions.maximum(my_params, 'wcss') + 10), step=10)) # podzialka na skali y
plt.xlabel('k', fontsize=14) # ustawienie labeli wykresu
plt.ylabel('WCSS', fontsize=14)
plt.show()
plt.scatter(data=my_params, x='k', y='iterations')
plt.plot('k', 'iterations', data=my_params)
plt.yticks(np.arange(functions.minimum(my_params, 'iterations'), functions.maximum(my_params, 'iterations')+1, step=1))
plt.xlabel('k', fontsize=14)
plt.ylabel('Liczba iteracji', fontsize=14)
plt.show()

# kNN
accuracy_scores = []
best_accuracy_score = 0
best_conf_matrix = []
best_k = 0
# wszystkie cechy
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k) # utworzenie klasyfikatora
    knn.fit(X_train, Y_train) # nauczenie klasyfikatora na zbiorze danych treningowych
    Y_pred = knn.predict(X_test) # sprawdzenie klasyfikatora na zbiorze danych testowych

    accuracy = accuracy_score(Y_test, Y_pred) # obliczenie dokladnosci klasyfikatora

    conf_matrix = confusion_matrix(Y_test, Y_pred) # obliczenie macierzy pomylek
    if accuracy > best_accuracy_score: # sprawdzenie czy aktualna dokladnosc klasyfikatora jest najlepsza
        best_accuracy_score = accuracy
        best_conf_matrix = conf_matrix
        best_k = k
    accuracy_scores.append((k, accuracy*100))


accuracy_scores = pd.DataFrame(accuracy_scores, columns=['num_of_neighbors','score'])
plt.bar(data=accuracy_scores, x='num_of_neighbors', height='score')
plt.xticks(np.arange(1, 16, step=1))
plt.yticks(np.arange(0, 110, step=10))
plt.xlabel('k', fontsize=14)
plt.ylabel('Dokładność klasyfikacji w %', fontsize=14)
plt.show()
print('wszystkie cechy')
print(best_conf_matrix)
print(best_accuracy_score)
print("k: "+str(best_k))
#pary cech
for x_label, y_label in combinations(df_train.columns[:-1], 2):
    accuracy_scores = []
    best_accuracy_score = 0
    best_conf_matrix = []
    print(names_in_polish_dict[x_label]+' '+names_in_polish_dict[y_label])
    for k in range(1,16):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train[[x_label, y_label]], Y_train) # trenowanie klasyfikatora na wylacznie 2 cechach
        Y_pred = knn.predict(X_test[[x_label, y_label]])
        accuracy = accuracy_score(Y_test, Y_pred)
        conf_matrix = confusion_matrix(Y_test, Y_pred)
        if accuracy > best_accuracy_score:
            best_accuracy_score = accuracy
            best_conf_matrix = conf_matrix
            best_k = k
        accuracy_scores.append((k, accuracy*100)) #*100 zeby bylo w procentach

    accuracy_scores = pd.DataFrame(accuracy_scores, columns=['num_of_neighbors', 'score'])
    plt.bar(data=accuracy_scores, x='num_of_neighbors', height='score')
    plt.yticks(np.arange(0, 110, step=10))
    plt.xticks(np.arange(1, 16, step=1))
    plt.xlabel('k', fontsize=14)
    plt.ylabel('Dokładność klasyfikacji w %', fontsize=14)
    plt.show()
    print(best_conf_matrix)
    print(best_accuracy_score)
    print("k: "+str(best_k))