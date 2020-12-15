
class Utility:
    pd = import pandas 
    np = import numpy 
    plt = import matplotlib.pyplot 
    #     %matplotlib inline
    sns = import seaborn 
    mpatches = import matplotlib.patches 
    plt.style.use('ggplot')
    
    
    def __init__(self,data_path):
        self.data_path = data_path
        
    def summarise(self,usecols):
        '''
        returns df
        '''
        global df
        if len(usecols)==0: 
            df = pd.read_csv(self.data_path)
        
        else:
            df = pd.read_csv(self.data_path,usecols=usecols)
#         print(df.columns)
        
        print('data types of all the columns are')
        #seeing dtypes
        dtype = df.dtypes
        print(dtype)
        print('\n')
        
        print('summary of the data')
        desc = df.describe()
        print(desc)
        print('\n')
        
        
        Nans = df.isnull().sum()
        print('all the null values in the columns: \n ',
              Nans)

        return df
    
    def plotter(self,*args ,hue_col=None,kind='kde',color='#bdbdbd'):
        '''
        1.first scatter columns
        2.second jointplot kind='kde' default---
        options for 'hex'|'scatter'| "reg" | "resid" 
        
        3.third countplot
        
        4.fourth heatmap of correlation no need to provide col names in args
        
        '''
        #scatter plot 
        plt.figure(figsize=(13,10))
        plt.scatter(args[0][0][0],args[0][0][1],data=df,cmap=hue_col,color=color)
        plt.xlabel(args[0][0][0])
        plt.ylabel(args[0][0][1])
        
        #jointplot to see the dist
#         plt.figure(figsize=(13,10))
        sns.jointplot(x=args[0][1][0],y=args[0][1][1],data=df,kind=kind,color=color)
        plt.xlabel(args[0][1][0])
        plt.ylabel(args[0][1][1])
        
        
        #countplot
        plt.figure(figsize=(13,10))
        sns.countplot(args[0][2][0],data=df,hue=hue_col,color=color)
       
        #heatmap
        
        plt.figure(figsize=(13,10))
        corr = df.corr()
        sns.heatmap(corr,cmap='BuGn_r')
        plt.show()