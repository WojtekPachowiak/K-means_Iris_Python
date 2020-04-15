import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from termcolor import colored
import os


df = pd.read_csv("C:/Users/Administrator/Desktop/iris.csv", names=["sepal_length", "sepal_width", "petal_length", "petal_width","class"])

#######################################################################################


def initialize_centroids(x_subset, y_subset, df):
    centroids = []
    for i in range(k):
        x_value = round(random.uniform(min(df[x_subset]),max(df[x_subset])),1)
        y_value = round(random.uniform(min(df[y_subset]),max(df[y_subset])),1)
        centroids.append([x_value,y_value])
    return centroids

#######################################################################################


def euclid_distance(df,centroids):
    distance = [[],[],[]]
    clustered_elements = [pd.DataFrame([]),pd.DataFrame([]),pd.DataFrame([])]

    for c in range(len(df)):
        for i in range(k):
            distance[i] = (df.iloc[[c]]-centroids[i])
            distance[i] = distance[i]**2
            distance[i] = float((distance[i].iat[0,0]+distance[i].iat[0,1])**(1/2))

        if (distance[0] < distance[1]) and (distance[0] < distance[2]):
             clustered_elements[0] = pd.concat([df.iloc[[c]], clustered_elements[0]], ignore_index=True)

        elif (distance[1] < distance[0]) and (distance[1] < distance[2]):
             clustered_elements[1] = pd.concat([df.iloc[[c]], clustered_elements[1]], ignore_index=True)

        elif (distance[2] < distance[1]) and (distance[2] < distance[0]):
             clustered_elements[2] = pd.concat([df.iloc[[c]], clustered_elements[2]], ignore_index=True)
        # input()
    return clustered_elements

#######################################################################################

def move_centroids(centroids, clustered_elements):

    for i in range(k):
        if len(clustered_elements[i]) != 0:
            centroids[i] = list(clustered_elements[i].sum()/len(clustered_elements[i]))    #assign new coordinates to the centroid
        else:
            centroids[i] = [round(random.uniform(min(df[x_subset]),max(df[x_subset])),1),  round(random.uniform(min(df[y_subset]),max(df[y_subset])),1)]   #assign random coordiantes to the lonely centroid

    return centroids


#######################################################################################

k = 3
colours = ["red","blue","green"]
answers = {"a":"sepal_length",  "b":"sepal_width",  "c":"petal_length",  "d":"petal_width"}



while True:
    ##################################Choosing subsets######################################
    answers_copy = dict(answers)

    print(colored("Iris dataset","yellow"),"\nChoose two subsets to classify.\n_______________________________\n")
    for x,y in answers.items():
        print(x,"\t",y)
    choice = input(colored("\nFirst subset: ","cyan"))
    x_subset = answers[choice]
    answers_copy.pop(choice)
    print("\n-------------------------------\n")
    for x,y in answers_copy.items():
        print(x,"\t",y)
    choice = input(colored("\nSecond subset: ","cyan"))
    y_subset = answers[choice]


    df1 = df[[x_subset, y_subset]]
    x_axis = df[x_subset]
    y_axis = df[y_subset]

    #########################################################################################

    centroids = initialize_centroids(x_subset, y_subset, df1)
    iterations =0

    while True:
        iterations+=1
        plt.clf()
        clustered_elements = euclid_distance(df1,centroids)

        centroids_prev_iteration = centroids[:] #control variable

        centroids = move_centroids(centroids, clustered_elements)

        if centroids_prev_iteration == centroids:   #check if centroids have moved #if they haven't: break loop
            break


        for i in range(k):
            plt.scatter(centroids[i][0],  centroids[i][1], color=colours[i],marker="o",zorder=1,s=100,edgecolors ="k")

        for i in range(len(clustered_elements[0])):
            plt.scatter(clustered_elements[0].loc[[i],x_subset],  clustered_elements[0].loc[[i],y_subset], color=colours[0],marker="^",s=70,zorder=-1,alpha =0.5)

        for i in range(len(clustered_elements[1])):
            plt.scatter(clustered_elements[1].loc[[i],x_subset],  clustered_elements[1].loc[[i],y_subset], color=colours[1],marker="^",s=70,zorder=-1,alpha =0.5)

        for i in range(len(clustered_elements[2])):
            plt.scatter(clustered_elements[2].loc[[i],x_subset],  clustered_elements[2].loc[[i],y_subset], color=colours[2],marker="^",s=70,zorder=-1,alpha =0.5)


        plt.xlabel('{} [cm]'.format(x_subset))
        plt.ylabel('{} [cm]'.format(y_subset))
        plt.title("Iteration {}".format(iterations))
        plt.pause(0.00001)

    os.system('cls')
    input("Press any button to continue...")
    os.system('cls')
