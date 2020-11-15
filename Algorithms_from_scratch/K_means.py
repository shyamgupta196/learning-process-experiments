'''
author=='shyamgupta196'
version used == 'python 3.8.3'
'''
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

class Kmean:
    # a __init__ method with 
    def __init__(self,data,columns,k=2,iterations=8):
        '''consists of all the attributes of the 
        Kmean algo.'''
        #libraries numpy , pandas                                                                           #agar me columns ko hata du to and only operate on the numerical columns of the data 
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
    
    def show(self,col_index=[0,1]):
        
        plt.scatter(x=self.data[self.columns[col_index[0]]],y=self.data[self.columns[col_index[1]]])
        plt.xlabel(self.columns[col_index[0]])
        plt.ylabel(self.columns[col_index[1]])
        plt.show()
    
        
    def fit(self):
        #seeing datapoints of columns n checking if everythng is int 
        #if dtype == 'object'
         #raise Error
        for i in self.columns:
            if self.data[i].dtype =='O':
                raise Error('data should not be categorical')
        
        #what if i could make a dict of my own which can save the combinations for every attempt 
        dict_of_means = {}
        dict_of_points  = {}
        point_count = 0
        cluster1 = {}
        cluster2 = {}
        
        
        #declaring k lists for k clusters
        list_point_sum1 = [[] for _ in range(self.iterations)]
        list_point_sum2 = [[] for _ in range(self.iterations)]
        #MULTI LISTS FOR XY SUM instead i took a dict to save the inputs 
        
#start a loop for 8 iterations or more 
        for iters in range(self.iterations):

            #take k random points 
            # for i in self.columns:
            #we code now only for 2 clusters
            arr_cols = np.array(self.data[self.columns])
            point1 = random.choice(arr_cols.tolist())
            # print(point1)
            point2 = random.choice(arr_cols.tolist())
            # print(point2)
            dict_of_points.update({point_count:[point1,point2]})
#now we have  a record of number of points taken to find the minkowski-dist

            #calculating SSE from the points 
            for i in range(len(self.data)):
#start calculating the Ed(euclidian distance) or normal for medoids
               
               
               '''
               this is just distance not the point 
               ye sirf 303 alag alag points ka dist hai 
               not the 303 points which are to be sent to clusters 
               
               we need a varible which consists of the points from which the distances are calculated 
               
               not done yrr 
               '''
               
               
               
                dist_1 = np.array(self.data.iloc[i,:2]) - point1
                dist_2 = np.array(self.data.iloc[i,:2]) - point2
                # print(dist_1)
                # print(dist_2)
                #now take sum of the xy
                
                sum_1 = dist_1[0]+dist_1[1]
                sum_2 = dist_2[0]+dist_2[1]
                # print(sum_1)
                # print('-'*20)
                
                # dist_1 = list(dist_1)
                if sum_1<sum_2:
                    # cluster1[iters].append(dist_1)
                    cluster1.update({point_count:dist_1})
                elif sum_2<sum_1:
                    # list_point_sum2[iters].append(dist_2)
                    cluster2.update({point_count:dist_2})
# scalars@@@@@@@danger coming array sum elements
            
            point_count+=1
            
# at last find the min value of the SSE list_no_(k) n its corresponsing index point will be the 
        # list_point_sum1 = sorted(list_point_sum1)
        # list_point_sum2 = sorted(list_point_sum2)
        
        #calculating the sum of the sorted lists        
        # list_list_sum1 = [sum(point_sum) for point_sum in list_point_sum1]        
        # print('point sum',list_point_sum1)
        # list_list_sum2 = [sum(point_sum) for point_sum in list_point_sum2]        
        # print('point sum',list_list_sum2)
        
        
        
        # why 303 elements not coming in every list 
        # for i in range(0,40):
        #     print(len(list_point_sum1[i]))
        # print(';'*20)
        # for i in range(0,40):
        #     print(len(list_point_sum2[i]))
        print(cluster1)
        
        print(cluster2)
        print(dict_of_points)
        # plt.plot(dict_of_points.val)
        # plt.scatter(cluster1.values(),cluster2.values())
        plt.show()
        
# optimized point for the cluster 
#finally no of clusters should be plotted 
            
            
        # print(list_point_sum1)
        # print(dict_of_points)
        
            
            

# if k not greater than no of points in the data 
 
#every point has k distances associated with it 
#so the one with argmin dist will append to that cluster list 
#calc. the SSE in of the points with the random point selected
'''make a seperate list to store the SSE values 
simultaneously storing every random point taken with it
in a list at same index  '''
#take mean of the clustered points saved in the lists 

#then repeat the process of taking the distance between the points  (for 8 or more iterations)

df = pd.read_csv('heart.csv')
inst = Kmean(data=df,columns=['thalach','chol'],iterations=4)


# inst.show()
inst.fit()


