import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix





class Utilis():
    def __init__(self):
        self.label_type="center"
        self.size = 12
        
        #self.column_values_1 =  X_train.Gender.value_counts()
        #self.column_values_2 =  X_train["Customer Type"].value_counts()
        #self.column_values_3 =  X_train["Type of Travel"].value_counts()
        
        self.details_attributs = ['Homme/Femme', 'Loyal/Disloyal','Age du client',
                                  'Objet du voyage (e.g. Personnel/Business)',
                                  'Eco/Eco Plus/Business Class','Distance parcourue',
                                  'Niveau de satisfaction du WIFI (0 à 5)',
                                  'Niveau de satisfaction Départ/Arrivée',
                                  'Niveau de satisfaction réservation en ligne',
                                  'Niveau de satisfaction "gate localisation"',
                                  'Niveau de satisfaction sur la nourriture et la boisson',
                                  'Niveau de satisfaction sur l"embarquement',
                                  'Niveau de satisfaction sur le confort des sièges',
                                  'Niveau de satisfaction sur les divertissements',
                                  'Niveau de satisfaction sur les services à bord',
                                  'Niveau de satisfaction sur l"espacement des jambes',
                                  'Niveau de satisfaction sur la prise en charge des baggages',
                                  'Niveau de satisfaction sur l"enregistrement',
                                  'Niveau de satisfaction sur les services durant le vol',
                                  'Propreté',
                                  'Retard (en minutes) depuis le tard',
                                  'Retard (en minutes) à l"arrivée',
                                  'variable à expliquer : Neutral or dissatisfaction/Satisfaction'
                                 ]
    def detail_attributs(self,dataframe):
        value = self.details_attributs
        table_attributs = pd.DataFrame(columns=['Column''s name',"Details"])
        for i in range(len(value)):
           
            table_attributs.loc[i,['Column''s name']] = dataframe.columns[i]
            table_attributs.loc[i,['Details']] = value[i]
        return table_attributs
    
    def missing_values (self,dataframe):
        table_pivot = pd.DataFrame(columns=['Column''s name',"# missing values"])
        for i in range(len(dataframe.columns)):
            table_pivot.loc[i,['Column''s name']] = dataframe.columns[i]
            table_pivot.loc[i,['# missing values']] = dataframe[dataframe.columns[i]].isnull().sum()
        return table_pivot
    
    def plot_annotation(self,axes,data):
        for c in axes.containers:
            labels = [f'{h/data.satisfaction.count()*100:0.1f}%' if (h := v.get_height()) > 0 else '' for v in c]
            axes.bar_label(c, labels=labels, label_type=self.label_type, size=self.size)
    
    def plot_hist(self,train,test):
        fig, axes = plt.subplots(1, 2,  figsize=(12,6))
        sns.set(style="darkgrid")
        fig.suptitle("Distribution de la variable satisfaction \npar base d'entraînement ou de test")
        sns.countplot(ax=axes[0],x="satisfaction", data=train)
        axes[0].set_title("base d'entraînement")
        sns.countplot(ax=axes[1],x="satisfaction", data=test)
        axes[1].set_title('base de test')
        Utilis().plot_annotation(axes[0],train)
        Utilis().plot_annotation(axes[1],test)
    
    def plot_pie(self,len_cat,categorical_cols,train):
        
        fig, axes = plt.subplots(1, len_cat,  figsize=(12,6))
        colors = sns.color_palette('pastel')
        for i in range(len_cat):
            cols = categorical_cols[i]
            column_values =  train[cols].value_counts()  
            #create pie chart
            axes[i].pie(column_values.values, labels = column_values.index, colors = colors, autopct='%.01f%%')
        plt.show()
        
    def define_cols(self,data):
        categorical_columns_selector = selector(dtype_include='category')
        categorical_columns = categorical_columns_selector(data)
        
        numerical_columns_selector = selector(dtype_exclude='category')
        numerical_columns = numerical_columns_selector(data)
        
        indices = 0, 1,2
        ordinal_cols = [i for j, i in enumerate(categorical_columns) if j not in indices]
        categorical_cols = [i for j, i in enumerate(categorical_columns) if j  in indices]
        return ordinal_cols, categorical_cols, numerical_columns
    
    
    def plot_group_features(self, variable,data):
        groupby_class = pd.DataFrame({'count' : data.groupby( [ "satisfaction", variable] ).size()}).reset_index()
        groupby_class["perc"] = 100 * groupby_class['count'] / groupby_class.groupby('satisfaction')['count'].transform('sum')
        fg = sns.catplot(x=variable, y="perc",col="satisfaction",palette="YlOrBr", data=groupby_class, kind="bar")
        fg.fig.subplots_adjust(top=0.9)
        for ax in fg.axes.ravel():
            # add annotations
            for c in ax.containers:
                labels = [f'{h:0.1f}%' if (h := v.get_height()) > 0 else '' for v in c]
                ax.bar_label(c, labels=labels, label_type='edge')
        plt.show()
    
    
    def plot_features(self,train,ordinal_cols):
        fig, axes = plt.subplots(5, 3,  figsize=(20,18))
        sns.set(style="darkgrid")
        count = 0
        for i in range(5):
            for j in range(3):
                sns.countplot(ax=axes[i][j],x=ordinal_cols[count], data=train, palette="Set3")
                for c in axes[i][j].containers:
                    labels = [f'{h/train[ordinal_cols[count]].count()*100:0.1f}%' if (h := v.get_height()) > 0 else '' for v in c]
                    axes[i][j].bar_label(c, labels=labels, label_type="center", size=12)
                    count+=1
        
            
            
        
        plt.show()  
    
    def categorical_preprocessor(self):
        
        categorical_preprocessor = Pipeline(
    steps=[
          ("onehot", OneHotEncoder(drop='first'))
          ]
                                   )
        return categorical_preprocessor

    def ordinal_encoder(self,df):
        
        categorical_indexes = [df.columns.get_loc(c) for c in df.columns]
        df.iloc[:,categorical_indexes] = df.iloc[:,categorical_indexes].astype("int64")
        return df   
    
    

    def ordinal_preprocessor(self,cats):
        
        ordinal_preprocessor = Pipeline(
    steps=[
          ("ordinalhot", OrdinalEncoder(categories=cats)),
          ])
        return ordinal_preprocessor

    def numerical_preprocessor(self):
        
        numerical_preprocessor = Pipeline(
    steps=[
          ("Impute", SimpleImputer(missing_values=np.nan, strategy='median')),
          ("Scaling",StandardScaler())
          ])
        return numerical_preprocessor

    def num_encoder(self):
        
        num_encoder = Pipeline(
  steps=[
        ('num', FunctionTransformer(self.ordinal_encoder))
        ]
                    )
        return num_encoder
    
    
    def pd_perc(self,df, col):
        
        vp =pd.DataFrame(df[col].value_counts().sort_index(), columns=[col])
        vp["perc"] = round((vp[col]/sum(vp[col]))*100,2)
        vp['index'] = vp.index
        return vp
    
        
    
    def sumzip(self, *items):
        return [sum(values) for values in zip(*items)]
    
    def plot_confusion_matrix(self,y_test, y_pred):
        
        cf_matrix = confusion_matrix(y_test, y_pred)
        group_names = ['True Neg','False Pos','False Neg','True Pos']
        group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
        group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
        labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        return cf_matrix, labels, group_counts
    
    def metrics_classifier(self,pipelines,models_names,train,y_train,test,y_test):
        
        
        
        
        name_models, f1_score_models_train, f1_score_models_test,accuracy_score_models_train, accuracy_score_models_test,\
        precision_score_models_train, precision_score_models_test,\
        recall_score_models_train, recall_score_models_test= [], [], [], [],[],[],[],[],[]
           
        for p, name in zip(pipelines, models_names):
            
            p.fit(train, y_train)
            y_pred_test = p.predict(test)
            y_pred_train = p.predict(train)
            name_models.append(name)
            f1_score_models_test.append("{:2.1f}%".format(f1_score(y_test, y_pred_test) * 100))
            f1_score_models_train.append("{:2.1f}%".format(f1_score(y_train, y_pred_train) * 100))
            
            accuracy_score_models_test.append("{:2.1f}%".format(accuracy_score(y_test, y_pred_test) * 100))
            accuracy_score_models_train.append("{:2.1f}%".format(accuracy_score(y_train, y_pred_train) * 100))
            
            precision_score_models_test.append("{:2.1f}%".format(precision_score(y_test, y_pred_test) * 100))
            precision_score_models_train.append("{:2.1f}%".format(precision_score(y_train, y_pred_train) * 100))
            
            recall_score_models_test.append("{:2.1f}%".format(recall_score(y_test, y_pred_test) * 100))
            recall_score_models_train.append("{:2.1f}%".format(recall_score(y_train, y_pred_train) * 100))
            
        return name_models, f1_score_models_train, f1_score_models_test, accuracy_score_models_test, accuracy_score_models_train,\
        precision_score_models_test, precision_score_models_train, recall_score_models_test, recall_score_models_train
    