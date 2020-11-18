# %%
'''
author=='shyamgupta196'
version used == 'python 3.8.3'
this is hardcoded only to handle 2 [numerical] features columns 
'''
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class Kmean:
    # a __init__ method with
    def __init__(self, data, columns, k=2, iterations=8):
        '''consists of all the attributes of the 
        Kmean algo.'''
        # libraries numpy , pandas                                         #agar me columns ko hata du to and only operate on the numerical columns of the data
        # but how to  make new clusters

        # number of clusters(K)
        # pandas df
        # 2-columns dtype=='list'
        self.data = data

        # if isinstance(self.columns,list):
        if type(columns) == type(list()):
            self.columns = columns
        else:
            raise Exception('list of columns to be provided')
        self.k = k
        self.iterations = iterations

    def show(self, col_index=[0, 1]):

        plt.scatter(x=self.data[self.columns[col_index[0]]],
                    y=self.data[self.columns[col_index[1]]])
        plt.xlabel(self.columns[col_index[0]])
        plt.ylabel(self.columns[col_index[1]])
        plt.show()

    def fit(self):
        # seeing datapoints of columns n checking if everythng is int
        # if dtype == 'object'
        #raise Error
        for i in self.columns:
            if self.data[i].dtype == 'O':
                raise Error('data should not be categorical')

        # what if i could make a dict of my own which can save the combinations for every attempt

        dict_of_points = {}
        point_count = 0
        cluster1 = {i: [] for i in range(self.iterations)}
        cluster2 = {i: [] for i in range(self.iterations)}

        # declaring k lists for k clusters
        list_point_sum1 = [[] for _ in range(self.iterations)]
        list_point_sum2 = [[] for _ in range(self.iterations)]
        # MULTI LISTS FOR XY SUM instead i took a dict to save the inputs

# start a loop for 8 iterations or more
        for iters in range(self.iterations):

            # take k random points
            # we code now only for 2 clusters
            arr_cols = np.array(self.data[self.columns])
            point1 = random.choice(arr_cols.tolist())
            # print(point1)
            point2 = random.choice(arr_cols.tolist())
            # print(point2)
            dict_of_points.update({point_count: [point1, point2]})
# now we have  a record of number of points taken to find the points-dist

            # cluster_points = 0
            
            # calculating dist from the points
            single_point = np.array(self.data.loc[:, self.columns])
            for i in single_point:
                # start calculating the (distance) or normal for medoids

               # '''
               # this is just distance not the point
               # ye sirf 303 alag alag points ka dist hai
               # not the 303 points which are to be sent to clusters

               # we need a varible which consists of the points from which the distances are calculated

               # not done yrr
               # '''

                dist_1 = i - point1
                
                dist_2 = i - point2
                # now take sum of the xy


                sum_1 = abs(dist_1[0])+abs(dist_1[1])
                sum_2 = abs(dist_2[0])+abs(dist_2[1])

                if sum_1 < sum_2:
                    # '''n
                    # this is not feaasible since every point gets assigned different value n hence we cannot use dictionary
                    # so we use alist to append all the values of a particular iteration n then
                    # map the values to the respective point of the dictionary
                    # cluster1.update({cluster_points:i.tolist()})
                    # '''

                    cluster1[iters].extend([i.tolist()])

                elif sum_2 < sum_1:

                    cluster2[iters].extend([i.tolist()])
        

            # taking sum of all the elements of the clusters 
            cluster1_list_sum_append = {i:[] for i in range(self.iterations)}
            
            cluster2_list_sum_append = {i:[] for i in range(self.iterations)}
            print(cluster1)
            
            for i in range(self.iterations):
                length  = len(cluster1[i])
                for j in range(length):
                    sums_1 = np.sum(list(cluster1.values()))[i][j]
                    sums_2 = np.sum(list(cluster2.values()))[i][j]
                    print(sums_1)
                    cluster1_list_sum_append[i].extend(sums_1)
                    cluster2_list_sum_append[i].extend(sums_2)
            print(cluster1_list_sum_append)
                        
            cluster1_sum = {i:sum(j for j in cluster1_list_sum_append[i]) for i in range(self.iterations)}
                 
            cluster2_sum = {i:[] for i in range(self.iterations)}
            


            point_count += 1
            
        print('clus1', cluster1)
        print("*"*20)
        print('clus2', cluster2)
        print(dict_of_points)


df = pd.read_csv('heart.csv')
inst = Kmean(data=df, columns=['age', 'trestbps'], iterations=4)

inst.fit()


#swap case sum compaarision for making something a cluster center
