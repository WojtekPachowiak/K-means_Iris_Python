
import math
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
import os

from pandas.core.frame import DataFrame
from sklearn import datasets


#======================
class K_means:
    k = None
    features_values_range = [[None, None],[None, None]]


    def init(k):
        K_means.k = k

    def _set_features_values_range(f_v_range):
        K_means.features_values_range = f_v_range


    def initialize_centroids(df_subset):
        centroids = []
        
        x_range = [min(df_subset.iloc[:,0]),max(df_subset.iloc[:,0])]
        y_range = [min(df_subset.iloc[:,1]),max(df_subset.iloc[:,1])]
        K_means._set_features_values_range([x_range,y_range])

        for i in range(K_means.k):
            x_value = random.uniform(x_range[0], x_range[1])
            x_value = round(x_value,1)
            y_value = random.uniform(y_range[0], y_range[1])
            y_value = round(y_value,1)
            centroids.append([x_value,y_value])
        return centroids


    def move_centroids(centroids, clustered_observations):
        prev_centroids = centroids[:]
        for i in range(K_means.k):
            if len(clustered_observations[i]) != 0:
                #assign new coordinates to the centroid
                centroids[i] = clustered_observations[i].mean(axis=0).round(1).values.tolist()

            else:
                #assign random coordiantes to the lonely centroid
                x = round(random.uniform(K_means.features_values_range[0][0], K_means.features_values_range[0][1]),1)
                y = round(random.uniform(K_means.features_values_range[1][0], K_means.features_values_range[1][1]),1)
                centroids[i] = [x, y]   
        is_end = False
        if prev_centroids == centroids:
            is_end = True
        return centroids, is_end

    def categorize_observations(observations, centroids):
        clustered_observations = [pd.DataFrame() for _ in range(len(centroids))]
        observations = observations.values.tolist()
        for point in observations:
            shortest_dist = math.inf
            closest_centroid_index = None
            for i in range(len(centroids)):
                dist = math.dist(point, centroids[i])
                if dist < shortest_dist: 
                    shortest_dist = dist
                    closest_centroid_index = i
            clustered_observations[closest_centroid_index] = clustered_observations[closest_centroid_index].append(pd.DataFrame([point]))
        return clustered_observations
#======================


#======================
class Visualization:
    # colours = ["red","blue","green"]
    colours = None
    iteration = 1

    def init(k):
        if k == 3:
            Visualization.colours = ["red","blue","green"]
        else:
            Visualization.colours = plt.cm.jet(np.linspace(0,1,k))

    def draw_and_wait(categorized_observations, features, k):
        cat_obs = categorized_observations

        df = pd.DataFrame()
        for i in range(k):
            new = pd.DataFrame(cat_obs[i])
            new[""] = Visualization.colours[i]
            df = pd.concat([df,new])

        for i in range(k):
            plt.scatter(centroids[i][0],  centroids[i][1], c=Visualization.colours[i],marker="o",zorder=1,s=100,edgecolors ="k")
        plt.scatter(df.iloc[:,0],  df.iloc[:,1], c=df.iloc[:,2],marker="^",s=70,zorder=-1,alpha =0.5)

        plt.xlabel('{} [cm]'.format(features[0]))
        plt.ylabel('{} [cm]'.format(features[1]))
        plt.title("Iteration {}".format(Visualization.iteration))
        plt.pause(0.00001)
        plt.clf()

        Visualization.iteration+=1
    
    def reset_iter_counter():
        Visualization.iteration = 1
#======================



#======================
class Menu:
    possible_answers = {}
    last_letter_ASCII = None

    def Init(features : list):
        num_of_answers = len(features)
        first_letter_ASCII = ord('a')
        last_letter_ASCII = first_letter_ASCII + num_of_answers - 1
        Menu.last_letter_ASCII = last_letter_ASCII

        for i in range(num_of_answers):
            letter = chr(first_letter_ASCII+ i)
            Menu.possible_answers[letter] = features[i]
        Menu.num_of_features = num_of_answers


    def choosing_data_subset() -> list[str]:
        answers_left = Menu.possible_answers.copy()
        subset = []

        print("Iris dataset","\nChoose two subsets to classify.\n_______________________________\n")

        #first choice
        for x,y in answers_left.items():
            print(x,"\t",y)
        choice = input("\nFirst subset: ")
        Menu._check_input_correctness(choice)
        subset.append(Menu.possible_answers[choice])
        answers_left.pop(choice)
        

        print("\n-------------------------------\n")

        #second choice
        for x,y in answers_left.items():
            print(x,"\t",y)
        choice = input("\nSecond subset: ")
        Menu._check_input_correctness(choice)
        subset.append(Menu.possible_answers[choice])

        return subset
    
    def _check_input_correctness(input):
        if type(input) != str or len(input) == 0 or ord(input) > Menu.last_letter_ASCII or ord(input) < ord('a'):
            raise Exception('ERROR::WRONG INPUT CHARACTER GIVEN!') 
#======================



#======================
class Data:
    features = None
    df = None

    def Init():
        df = datasets.load_iris(return_X_y=False, as_frame = True)
        Data.df = df.data
        Data.features = df.feature_names
#======================












#######################################################################################
################################### APPLICATION #######################################
#######################################################################################

#centroids number
k = 3
K_means.init(k)
#load dataset and fill fields' values
Data.Init()
#tell Menu object what features are available for the user to choose from
Menu.Init(Data.features)
#specify how many colors need to be generated for visualization purposes
Visualization.init(k)

while True:
    #show menu, wait for user input and return user's desired dataset's subset
    df_subset_features = Menu.choosing_data_subset()
    #initialize K-means' centroids
    df_subset = Data.df[df_subset_features]
    centroids = K_means.initialize_centroids(df_subset)
    #reset the counter indicating which iteration is currently visualized
    Visualization.reset_iter_counter()
    
    while True:
        categorized_observations = K_means.categorize_observations(df_subset, centroids)
        centroids, is_end = K_means.move_centroids(centroids, categorized_observations)
        if is_end:
            break
        else:
            Visualization.draw_and_wait(categorized_observations, df_subset_features, K_means.k)

    os.system('cls')
    input("The K-means algorithm has converged. Press any button to continue...")
    os.system('cls')
