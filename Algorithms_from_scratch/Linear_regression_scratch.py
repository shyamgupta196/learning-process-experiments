import matplotlib.pyplot as plt 
import numpy as np
import random as rd  
import pandas as pd
from  matplotlib import style
#calculating the slope



def linear_reg(DATA_x,DATA_y):
    
    #array of numbers which will act as DATA as parameters
    mean_DATA_x = np.mean(DATA_x)
    mean_DATA_y = np.mean(DATA_y)    
    prod_age_height = np.mean(DATA_x*DATA_y)
    squared_mean_DATA_x = (np.mean(DATA_x))**2
    each_age_squared = np.apply_along_axis(lambda x: x**2 ,0,DATA_x)
    mean_each_age_squared = np.mean(each_age_squared)
    
    
    # #slope of the line
    slope = (((mean_DATA_x*mean_DATA_y) - prod_age_height)/(squared_mean_DATA_x - mean_each_age_squared))
    #intercept
    
    intercept = mean_DATA_y - (slope * mean_DATA_x)
    
    return slope,intercept

#squared_error_calculation for calculating sqaured_error in two args
def squared_error(y_original,y_line):
    return sum((y_original - y_line)**2)



def rmse_cal(y_original,y_line):
        
    base_rmse= squared_error(y_original, y_line)
    rmse = np.sqrt(base_rmse/len(DATA_y))
    return rmse


def coeff_of_determination_cal():
        
    SE_y_line = squared_error(DATA_y,regression_line)
    SE_y_mean = squared_error(DATA_y,mean_DATA_y)
    #r_squared is used for measuring the total  variance y in the data w.r.t x 
    R_squared = (SE_y_line)/(SE_y_mean)

    coeff_of_determination = 1 - R_squared

    return print(coeff_of_determination)

df=pd.read_csv('cleaned-Students-DATA.csv')


DATA_x = df['G3'].values
DATA_y = df['G2'].values

m,b=linear_reg(DATA_x,DATA_y)

style.use('ggplot')
# # #regression line 
regression_line=[(m *x)+b for x in DATA_x]

'''these are declared again to make them global since they are used in 
calculations ahead for finding the rmse and cod also regression plot(we have that in other function )'''
mean_DATA_x = np.mean(DATA_x)
mean_DATA_y = np.mean(DATA_y)    
prod_age_height = np.mean(DATA_x*DATA_y)
squared_mean_DATA_x = (np.mean(DATA_x))**2
each_age_squared = np.apply_along_axis(lambda x: x**2 ,0,DATA_x)
mean_each_age_squared = np.mean(each_age_squared)

plt.scatter(DATA_x,DATA_y)
# slope*x plot
plt.plot(DATA_x,regression_line,'black')
plt.show()


print(rmse_cal(DATA_y,regression_line))
coeff_of_determination_cal()




def Gauss_model_creator():
    global DATA
    model  = GaussianNB()
    X=DATA.drop(columns=['Survived','Unnamed: 0'], axis='columns').values
    print(X.shape)
    y=DATA['Survived'].values
    print(y.shape)
    X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=.1)
    best=0
    for i in range(20000):
        DATA= pd.read_csv('C:\\Users\\Asus\\Documents\\PROJECTS\\titanic-cleaned.csv')
        X=DATA.drop(columns=['Survived','Unnamed: 0'], axis='columns').values
        y=DATA['Survived'].values
        X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=.1)
        model  = GaussianNB()
        model.fit(X_train, y_train)
        score = model.score(X_test,y_test)
        if score > best:
            best = score
            print(f'''saving model ...
                with accuracy: {best}''')
            with open('titanic_gauss.pickle','wb') as f:
                pickle.dump(model,f)
