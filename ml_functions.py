import math
import random


class MyKMeans:

    def __init__(self, data, k, max_iterations=300):
        self.data = data # dane na ktorych bedzie przeprowadzana klasteryzacja
        self.k = k # liczba klastrow
        self.max_iterations = max_iterations # maksymalna liczba iteracji
        self.centroids = [] # lista centroidow
        self.num_of_iterations = 0 # liczba iteracji z jaka zakonczyla sie klasteryzacja
        self.labels = [0 for _ in range(len(self.data))]  # lista w ktorej znajduja sie etykiety wektorow, tworzy liste z samymi 0 o dlugosci rownej liczbie danych
        self.wcss = 0 # miejsce na parametr wcss

    def dist(self, vector1, vector2): # funkcja obliczajaca euklidesowy dystans pomiedzy wektorami a centroidami
        tmp_dist = 0
        for i in range(len(vector1)): # petla dodajaca do zmiennej kwadrat roznicy kazdej poszczegolnej cechy
            tmp_dist += (vector1[i] - vector2[i]) ** 2
        return math.sqrt(tmp_dist)

    def initial_centroids(self): # funkcja wybierajaca poczatkowe centroidy jako losowe wektory cech
        if len(self.data) >= self.k: # sprawdzenie czy liczba klastrow nie jest wieksza niz liczba danych
            used_indexes = {} # slownik na uzyte indeksy, zeby ten sam wektor nie zostal wybrany ponownie
            i=0
            while i < self.k: # petla wybierajaca losowo k wektorow cech i dodajaca je do listy centroidow
                random_index = random.randint(0, len(self.data) - 1)  # -1 bo inclusive z obu stron
                if random_index not in used_indexes: # sprawdzenie czy wektor o wylosowanym indeksie zostal juz uzyty
                    i += 1
                    used_indexes[random_index] = True # dodanie indeksu do uzytych indeksow
                    random_vector = self.data.iloc[random_index] # znalezienie wektora po indeksie w danych
                    self.centroids.append(tuple(random_vector)) # dodanie wektora do listy jako tuple

    def assign_centroids(self):  # funkcja przyporzadkowujaca wektory do klastrow, zwraca bool czy nastapily jakies zmiany
        changes = False # zmienna dzieki ktorej wiemy czy klasteryzacja sie zakonczyla
        for vector in self.data.itertuples(name=None): # iteruje po wektorach cech, kazdy wektor zwraca jako tuple z indeksem i kolejno wypisanymi cechami
            vector_index = vector[0] # zmienna na indeks wektora
            vector_centroid = self.labels[vector_index] # zmienna na indeks centroidu przyporzadkowanego do wektora
            min_dist = float('inf') # nieskonczonosc, napisane po to aby warunek sprawdzajacy czy cur_dist < min_dist zawsze przechodzil przy pierwszej iteracji dla danego wektora
            min_dist_centroid_index = 0 # zmienna na indeks najblizszego centroidu
            for index, centroid in enumerate(self.centroids): # petla iterujaca po kazdym centroidzie, dzieki enumerate mamy jeszcze indeks centroidu
                cur_dist = self.dist(vector[1:], centroid)  # obciecie indeksu
                if cur_dist < min_dist: # sprawdzenie czy policzony dystans jest mniejszy od aktualnego
                    min_dist = cur_dist
                    min_dist_centroid_index = index
            if min_dist_centroid_index != vector_centroid: # sprawdzenie czy wektor zostal przyporzadkowany do innego klastra
                self.labels[vector_index] = min_dist_centroid_index
                changes = True
        return changes

    def calc_centroids(self): # funkcja aktualizujaca polozenie centroidow
        num_of_attributes = len(self.data.columns) # liczba cech
        new_centroids = [list(0 for _ in range(num_of_attributes)) for _ in range(self.k)] # lista na nowe centroidy skladajaca sie z k centriodow wypelnionych zerami
        centroids_counts = [0 for _ in range(self.k)] # lista na rozmiary klastrow
        for vector in self.data.itertuples(name=None): # petla dodajaca wartosci wektorow cech poszczegolnych klastrow
            vector_index = vector[0]
            centroid_index = self.labels[vector_index]
            cur_centroid = new_centroids[centroid_index] # centroid aktualnego wektora
            centroids_counts[centroid_index]+=1 # aktualizacja rozmiaru klastrow
            # dodawanie
            for i in range(len(cur_centroid)):
                cur_centroid[i] += vector[i+1] # ominiecie indeksu
        # petla dzielaca kazda wartosc w centroidzie przez rozmiar odpowiadajacego klastra, dzieki temu otrzymujemy srednia arytmetyczna punktow nalezacych do klastrow
        for centroid_index in range(len(new_centroids)):
            cur_centroid = new_centroids[centroid_index]

            if cur_centroid == [0 for _ in range(num_of_attributes)]: # sprawdzenie czy klaster nie jest pusty
                new_centroids[centroid_index] = list(self.data.iloc[random.randint(0, len(self.data)-1)]) # wybranie nowego losowego centroidu
                continue # przeskoczenie do nastepnej iteracji
            # dzielenie
            for i in range(len(cur_centroid)):
                cur_centroid[i] /= centroids_counts[centroid_index]

        self.centroids = [tuple(centroid) for centroid in new_centroids]

    def run(self):
        self.initial_centroids() # wybranie poczatkowych centroidow
        for i in range(self.max_iterations):
            changes = self.assign_centroids() # przyporzadkowanie wektorow do klastrow
            self.calc_centroids() # aktualizacja polozenia centroidow
            if not changes: # sprawdzenie czy warunek stopu zostal spelniony
                self.num_of_iterations = i+1 # pierwsza iteracja to 0
                break
        self.calc_wcss() # obliczenie wcss
    def calc_wcss(self):
        sum_of_sq_dists = 0
        for vector in self.data.itertuples(name=None): # iterowanie po wektorach cech
            centroid_index = self.labels[vector[0]]
            sum_of_sq_dists += self.dist(vector[1:], self.centroids[centroid_index])**2
        self.wcss = sum_of_sq_dists