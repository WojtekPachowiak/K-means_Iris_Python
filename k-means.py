import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from termcolor import colored
import os


dane = pd.read_csv("C:/Users/Administrator/Desktop/iris.csv", names=["sepal_length", "sepal_width", "petal_length", "petal_width","class"])





#######################################################################################


def losowanie_centroidow(x_do_analizy, y_do_analizy, dane):
    centroidy = []
    for i in range(k):
        lenght = round(random.uniform(min(dane[x_do_analizy]),max(dane[x_do_analizy])),1)
        width = round(random.uniform(min(dane[y_do_analizy]),max(dane[y_do_analizy])),1)
        centroidy.append([lenght, width])
    return centroidy

#######################################################################################


def euclid_distance(dane,centroidy):
    centroid1 = pd.DataFrame([])
    centroid2 = pd.DataFrame([])
    centroid3 = pd.DataFrame([])
    for c in range(len(dane)):
        distance1 = (dane.iloc[[c]]-centroidy[0])
        distance2 = (dane.iloc[[c]]-centroidy[1])
        distance3 = (dane.iloc[[c]]-centroidy[2])

        distance1 = distance1**2
        distance2 = distance2**2
        distance3 = distance3**2

        distance1 = (distance1.iat[0,0]+distance1.iat[0,1])**(1/2)
        distance2 = (distance2.iat[0,0]+distance2.iat[0,1])**(1/2)
        distance3 = (distance3.iat[0,0]+distance3.iat[0,1])**(1/2)

        if (distance1 < distance2) and (distance1 < distance3):
             centroid1 = pd.concat([dane.iloc[[c]], centroid1], ignore_index=True)

        elif (distance2 < distance1) and (distance2 < distance3):
             centroid2 = pd.concat([dane.iloc[[c]], centroid2], ignore_index=True)

        elif (distance3 < distance2) and (distance3 < distance1):
             centroid3 = pd.concat([dane.iloc[[c]], centroid3], ignore_index=True)
        # input()

    return centroid1, centroid2, centroid3


#######################################################################################

def przesun_centroidy(centroidy, centroid1, centroid2, centroid3):
    if len(centroid1) != 0:
        centroidy[0] = list(centroid1.sum()/len(centroid1))
    else:
        centroidy[0] = [round(random.uniform(min(dane[x_do_analizy]),max(dane[x_do_analizy])),1),  round(random.uniform(min(dane[y_do_analizy]),max(dane[y_do_analizy])),1)]

    if len(centroid2) != 0:
        centroidy[1] = list(centroid2.sum()/len(centroid2))
    else:
        centroidy[1] = [round(random.uniform(min(dane[x_do_analizy]),max(dane[x_do_analizy])),1),  round(random.uniform(min(dane[y_do_analizy]),max(dane[y_do_analizy])),1)]

    if len(centroid3) != 0:
        centroidy[2] = list(centroid3.sum()/len(centroid3))
    else:
        centroidy[2] = [round(random.uniform(min(dane[x_do_analizy]),max(dane[x_do_analizy])),1),  round(random.uniform(min(dane[y_do_analizy]),max(dane[y_do_analizy])),1)]

    return centroidy


#######################################################################################


k = 3
colours = ["red","blue","green"]

while True:
    print(colored("Iris dataset","yellow"),"\nWybierz dwa zbiory cech, które posłużą do klasyfikacji irysów!\nPierwszy zbiór:")
    print("a \t sepal_length")
    print("b \t sepal_width")
    print("c \t petal_length")
    print("d \t petal_width")
    wybor = input('\n')

    if wybor == 'a':
        x_do_analizy = "sepal_length"
        print("Drugi zbiór:")
        print("b \t sepal_width")
        print("c \t petal_length")
        print("d \t petal_width")
        wybor = input('\n')

        if wybor == 'b':
            y_do_analizy = "sepal_width"
        elif wybor == 'c':
            y_do_analizy = "petal_length"
        elif wybor == 'd':
            y_do_analizy = "petal_width"

    elif wybor == 'b':
        x_do_analizy = "sepal_width"
        print("Drugi zbiór:")
        print("a \t sepal_length")
        print("c \t petal_length")
        print("d \t petal_width")
        wybor = input('\n')

        if wybor == 'a':
            y_do_analizy = "sepal_length"
        elif wybor == 'c':
            y_do_analizy = "petal_length"
        elif wybor == 'd':
            y_do_analizy = "petal_width"

    elif wybor == 'c':
        x_do_analizy = "petal_length"
        print("Drugi zbiór:")
        print("a \t sepal_length")
        print("b \t sepal_width")
        print("d \t petal_width")
        wybor = input('\n')

        if wybor == 'a':
            y_do_analizy = "sepal_length"
        elif wybor == 'b':
            y_do_analizy = "sepal_width"
        elif wybor == 'd':
            y_do_analizy = "petal_width"

    elif wybor == 'd':
        x_do_analizy = "petal_width"
        print("Drugi zbiór:")
        print("a \t sepal_length")
        print("b \t sepal_width")
        print("c \t petal_length")
        wybor = input('\n')

        if wybor == 'a':
            y_do_analizy = "sepal_length"
        elif wybor == 'b':
            y_do_analizy = "sepal_width"
        elif wybor == 'c':
            y_do_analizy = "petal_length"

#######################################################################################

    dane1 = dane[[x_do_analizy, y_do_analizy]]
    x_axis = dane[x_do_analizy]
    y_axis = dane[y_do_analizy]
    liczba_iteracji =0

    centroidy = losowanie_centroidow(x_do_analizy, y_do_analizy, dane1)

    while True:
        liczba_iteracji+=1
        plt.clf()
        centroid1, centroid2, centroid3 = euclid_distance(dane1,centroidy)
        centroid_prev_iteracja = centroidy[:]
        centroidy = przesun_centroidy(centroidy, centroid1, centroid2, centroid3)

        if centroid_prev_iteracja == centroidy:
            break

        for i in range(k):
            plt.scatter(centroidy[i][0],  centroidy[i][1], color=colours[i],marker="o",zorder=1,s=100,edgecolors ="k")

        for i in range(len(centroid1)):
            plt.scatter(centroid1.loc[[i],x_do_analizy],  centroid1.loc[[i],y_do_analizy], color=colours[0],marker="^",s=70,zorder=-1,alpha =0.5)

        for i in range(len(centroid2)):
            plt.scatter(centroid2.loc[[i],x_do_analizy],  centroid2.loc[[i],y_do_analizy], color=colours[1],marker="^",s=70,zorder=-1,alpha =0.5)

        for i in range(len(centroid3)):
            plt.scatter(centroid3.loc[[i],x_do_analizy],  centroid3.loc[[i],y_do_analizy], color=colours[2],marker="^",s=70,zorder=-1,alpha =0.5)
        # Note that using time.sleep does *not* work here!
        plt.xlabel('{} [cm]'.format(x_do_analizy))
        plt.ylabel('{} [cm]'.format(y_do_analizy))
        plt.title("Iteracja {}".format(liczba_iteracji))
        plt.pause(0.00001)


    os.system('cls')
    input("\nKliknij cokolwiek, aby kontynuować!")
    os.system('cls')
