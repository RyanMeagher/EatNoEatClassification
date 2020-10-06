#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.utils import shuffle
from sklearn import svm
from sklearn import tree
from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
#get_ipython().run_line_magic('matplotlib', 'inline')


# %%


# ran from the root directory of the project folder from the zip file provided 
import os;
cwd = os.getcwd()
print(cwd)


# %%


def GetFilePath(cwd):
    emg=[]
    groundTruth=[]
    imu=[]
    
    for dirName, subdirList, fileList in os.walk(cwd):

        if 'Truth' in dirName and len(fileList)>0 and 'txt' in fileList[0]:
            groundTruth.append(dirName+'/'+fileList[0])

        MyoData=[(dirName,file) for file in fileList if 'EMG' in file]

        for f in MyoData:

            emg.append(f[0]+'/'+f[1])

        MyoData=[(dirName,file) for file in fileList if 'IMU' in file]

        for f in MyoData:

            imu.append(f[0]+'/'+f[1])
    return emg, groundTruth, imu
    
emg, groundTruth, imu= GetFilePath(cwd)


# %%


def indexingFilePath(groundTruth,emg):
    user_idx_groundTruth=[]
    user_idx_myo=[]
    matches=[]
    
    for i in range(len(groundTruth)):
        user_idx_groundTruth.append(groundTruth[i].split('/')[-3:-1])

        user_idx_myo.append(emg[i].split('/')[-3:-1])
    for i in range(len(user_idx_myo)):
        if user_idx_myo[i][0]=='user09':
            user_idx_myo[i][0]='user9'



    for i in range(len(user_idx_groundTruth)):

        for j in range (len(user_idx_myo)):

            if user_idx_groundTruth[i]==user_idx_myo[j]:
                matches.append((user_idx_groundTruth[i][0],user_idx_groundTruth[i][1],i,j))
    return matches

matches=indexingFilePath(groundTruth,emg)


# %%


def CreateGroundTruthDict(matches):
    groundTruth_dict={}
    groundTruth_col_names=['Start','End','unknown']
    for item in matches:

        user_id,user_utensil,groundTruth_idx,myo_idx =item
        df_name=user_id+'_'+user_utensil


        df=pd.read_csv(groundTruth[groundTruth_idx],
                       names=groundTruth_col_names)


        groundTruth_dict[df_name]=df

    # there was an error in user19_spoon data there was a ' ' instead of ',' so I was unable to do 100/30 and then
    # make then round and make it type int64
    groundTruth_dict['user19_spoon'].iloc[4,:]=[771,794,1]
    groundTruth_dict['user19_spoon']=groundTruth_dict['user19_spoon'].astype('int64')


    # multiply start and end rows by 100/30 so that you obtain the sample rows. Since python is 0
    # based so we also need to subtract 1 off this number to get the proper starting and ending rows indexes
    # if we were working in matlab we would not need to do this as it is 1 based. 

    for df in groundTruth_dict.values():
        df['Start']=df['Start']*100/30-1
        df['Start']=df['Start'].round()
        df['Start']=df['Start'].astype('int32')

        df['End']=df['End']*100/30
        df['End']=df['End'].round()
        df['End']=df['End'].astype('int32')
    return groundTruth_dict


groundTruth_dict=CreateGroundTruthDict(matches)


# %%


def CreateUserDict(file_list,matches,col_names):
    user_dict={}
    for item in matches:
        user_id,user_utensil,groundTruth_idx,myo_idx =item
        df_name=user_id+'_'+user_utensil
        df=pd.read_csv(file_list[myo_idx],names=col_names)
        user_dict[df_name]=df
    return user_dict


# %%


emg_col_names=['timestamp','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8']
imu_col_names=['timestamp', 'Orientation X', 'Orientation Y', 'Orientation Z', 
               'Orientation W', 'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 
               'Gyroscope X', 'Gyroscope Y','Gyroscope Z']

emg_dict=CreateUserDict(emg,matches,emg_col_names)
imu_dict=CreateUserDict(imu,matches,imu_col_names)



# %%


#function to put a users actions into either eating or non-eating catagory
def findActionsUser(groundTruth,myo,col_names):
    eating=pd.DataFrame(columns=col_names)
    non_eating=pd.DataFrame(columns=col_names)
    start_eating_idx=0
    stop_eating_idx=0
    
    for i in range(len(groundTruth)-1): 
        start_eating_idx=groundTruth.loc[i, "Start"]
        stop_eating_idx=groundTruth.loc[i, "End"]
        next_start_eating_idx=groundTruth.loc[i+1, "Start"]
        next_stop_eating_idx=groundTruth.loc[i+1, "End"]
        
        #print(start_eating_idx,stop_eating_idx)

        eating=eating.append(myo.iloc[start_eating_idx:stop_eating_idx,:])
        if i==0:
            non_eating=non_eating.append(myo.iloc[:start_eating_idx,:])
        else:
            non_eating=non_eating.append(myo.iloc[stop_eating_idx:next_start_eating_idx,:])
            
        #print(eating)
        #print(non_eating)
    eating=eating.append(myo.iloc[next_start_eating_idx:next_stop_eating_idx,:])
    non_eating=non_eating.append(myo.iloc[next_stop_eating_idx:,:])
    
    return eating, non_eating


# %%


def getAllActions(groundTruth_dict,myo_dict,col_names,random_sampling=True):
    eating=pd.DataFrame(columns=col_names)
    non_eating=pd.DataFrame(columns=col_names)
    user_action_dict={}
    for user_id,df in groundTruth_dict.items():

    
        eat_user,not_eat_user=findActionsUser(groundTruth_dict[user_id],myo_dict[user_id],col_names)
        eat_user.drop("timestamp", axis=1, inplace=True)
        not_eat_user.drop("timestamp", axis=1, inplace=True)
        

        #print(user_id)
        #create a dict with individual user actions 

        
        eat_user['class']='eat'
        not_eat_user['class']='not_eat'


        eating=eating.append(eat_user, sort=True)
        if random_sampling:
            non_eating=non_eating.append(not_eat_user.sample(len(eat_user),random_state=10),sort=True)
            user_action_dict[user_id]=[eat_user,not_eat_user.sample(len(eat_user),random_state=10)]
            
            
        else:
            non_eating=non_eating.append(not_eat_user, sort=True)
            user_action_dict[user_id]=[eat_user,not_eat_user]
           
                
    return eating,non_eating,user_action_dict


eat_emg,not_eat_emg,emg_action_dict=getAllActions(groundTruth_dict,emg_dict,emg_col_names)
eat_imu,not_eat_imu,imu_action_dict=getAllActions(groundTruth_dict,imu_dict,imu_col_names)


# %%


# reset indexes of all individual users actions as well as the emg and imu eat/not_eat dataframes

def reset_index(m):
    rang=range(0,len(m))
    m.index.name = None
    m.index=rang
    return

reset_index(eat_emg)
reset_index(not_eat_emg)
reset_index(eat_imu)
reset_index(not_eat_imu)

for k in emg_action_dict.keys():
    reset_index(emg_action_dict[k][0])
    reset_index(emg_action_dict[k][1])
    reset_index(imu_action_dict[k][0])
    reset_index(imu_action_dict[k][1])
    
  


# %%


def eat_not_eat_emg_statistics(eat,not_eat):
    
    not_eat_feat=not_eat.iloc[:,:-1]
    not_eat_statistics=pd.DataFrame()
    not_eat_statistics['mean'] = not_eat_feat.mean(axis=1)
    not_eat_statistics['rms'] =np.mean((not_eat_feat.iloc[:,:]**2),axis=1).pow(1/2)
    not_eat_statistics['std']=not_eat_feat.std(axis=1)
    not_eat_statistics['max']=not_eat_feat.max(axis=1)
    not_eat_statistics['min']=not_eat_feat.min(axis=1)
    not_eat_statistics['range']=not_eat_feat.max(axis=1)-not_eat_feat.min(axis=1)
    not_eat_statistics['variance']=not_eat_feat.var(axis=1)
    not_eat_statistics['cumsum']=not_eat_feat.abs().sum(axis=1)
    not_eat_statistics['eucladian norm']=np.sqrt(np.square(not_eat_feat).sum(axis=1))
    not_eat_statistics['class']='not eat'
    
    eat_feat=eat.iloc[:,:-1]
    eat_statistics=pd.DataFrame()
    eat_statistics['mean'] = eat_feat.mean(axis=1)
    eat_statistics['rms'] = np.mean((eat_feat.iloc[:,:]**2),axis=1).pow(1/2)
    eat_statistics['std']=eat_feat.std(axis=1)
    eat_statistics['max']=eat_feat.max(axis=1)
    eat_statistics['min']=eat_feat.min(axis=1)
    eat_statistics['range']=eat_feat.max(axis=1)-eat_feat.min(axis=1)
    eat_statistics['variance']=eat_feat.var(axis=1)
    eat_statistics['cumsum']=eat_feat.abs().sum(axis=1)
    eat_statistics['eucladian norm']=np.sqrt(np.square(eat_feat).sum(axis=1))
    eat_statistics['class']='eat'
    
    
    return eat_statistics, not_eat_statistics


# %%


# creating dictionarys containing each users statistical_feature_matrix of [eating,non_eating]

emg_action_dict_feature_matrix={}
imu_action_dict_feature_matrix={}

for k in emg_action_dict.keys():
    emg_eat,emg_no_eat=eat_not_eat_emg_statistics(emg_action_dict[k][0],emg_action_dict[k][1])
    emg_action_dict_feature_matrix[k]=[emg_eat,emg_no_eat]
    
    imu_eat,imu_no_eat=eat_not_eat_emg_statistics(imu_action_dict[k][0],imu_action_dict[k][1])
    imu_action_dict_feature_matrix[k]=[imu_eat,imu_no_eat]
emg_action_dict_feature_matrix


# %%


#because this is time series take aggregation to find the mean of each of the 
#5 statistical values over the amount of rows that allows we feed into row_merge. This allows us to have equal 
#number of eating and non-eating values between all the users. This can also be applied over time series that is 
#gathered in the future.



def row_merge(df_dict,n):
    aggregated_emg={}
    for k,v in df_dict.items():
        usr_eat, usr_not_eat=v[0],v[1]
        usr_len=v[0].shape[0]
        idx=np.linspace(0,usr_len,n)
        idx=idx.astype(int)
        df_eat=pd.DataFrame()
        df_not_eat=pd.DataFrame()



        for i in range(n-1):
            start=idx[i]
            stop=idx[i+1]
            df_eat=df_eat.append(usr_eat.iloc[start:stop,:-1].mean(axis=0), ignore_index=True)
            df_not_eat=df_not_eat.append(usr_not_eat.iloc[start:stop,:-1].mean(axis=0), ignore_index=True)
        df_eat['class']='eat'
        df_not_eat['class']='not eat'
        aggregated_emg[k]=[df_eat,df_not_eat]
    return aggregated_emg
    

        
        
aggregated_emg=row_merge(emg_action_dict_feature_matrix,50)
        
        
aggregated_emg


# %%


def append_df(df1,df2):
    return df1.append(df2, ignore_index=True)

merged_emg_feature_matrix={}
merged_aggregated_emg={}

for k,v in emg_action_dict_feature_matrix.items():
    merged_emg_feature_matrix[k]=append_df(emg_action_dict_feature_matrix[k][0],
                                           emg_action_dict_feature_matrix[k][1])
    
    merged_aggregated_emg[k]=append_df(aggregated_emg[k][0],
                                       aggregated_emg[k][1])
                                           
merged_emg_feature_matrix
merged_aggregated_emg
    


# %%


type(list(emg_action_dict_feature_matrix.values())[0][0])


# %%


#phase 1 train test split so that 60% of each user is in training set and 40% of each user is in testing set
# we will merge these together and apply pca over whole matrix then we will split them up again at the index 
# that represents the 60%. In doing this we will be able to run pca on all data together and then split it up
# at the appropriate point so as to gurantee that we have 60% of each users data in the training set and 40% of
#each users data in the test set.

#we have shuffled and stratified so that the samples are choosen randomly for test and train and so that 
# the test and train contain a 50/50 ratio of eat/no eat samples

def create_60_40_user_dependent_train_test(merged_emg_feature_matrix):
    from sklearn.model_selection import train_test_split
    
    train=pd.DataFrame()
    test=pd.DataFrame()

    for k,df in merged_emg_feature_matrix.items():
        
        X_user=df.iloc[:,:-1]
        y_user=df['class']

        X_train, X_test, y_train, y_test = train_test_split(X_user, y_user, test_size=0.4,train_size=0.6,random_state=42,
                                                            shuffle=True,stratify=y_user)
        train_user = pd.concat([X_train, y_train], axis=1)
        test_user= pd.concat([X_test, y_test], axis=1)
        
        train=train.append(train_user)
        test=test.append(test_user)

        
    return train,test ,train.shape[0] 

            
            
train_dependent,test_dependent, idx_60_percent_dependent=create_60_40_user_dependent_train_test(merged_aggregated_emg)



comined_train_test_dependent=append_df(train_dependent,test_dependent)
reset_index(comined_train_test_dependent)




comined_train_test_dependent


# %%


train_dependent['class'].value_counts(), test_dependent['class'].value_counts()


# %%


# phase 2 we prepare our datasets so that we split the data so that in our training set we have 18 users spoon 
# and fork actions and in the test we have 12 different users spoon and fork actions. 
def create_60_40_user_independent_train_test(merged_emg_feature_matrix):
    from sklearn.model_selection import train_test_split
    
    train=pd.DataFrame()
    test=pd.DataFrame()
    count=0
    idx_60_percent=int(len(merged_emg_feature_matrix.keys())*.6)
    
    for k,df in merged_emg_feature_matrix.items():
        count+=1
        
        
        if count <=idx_60_percent:
            train=train.append(df)
            
        
        else:
            test=test.append(df)
        

        
    return train,test, train.shape[0] 


train_independent, test_independent, idx_60_percent_independent=create_60_40_user_independent_train_test(merged_aggregated_emg)

comined_train_test_independent=append_df(train_independent,test_independent)
reset_index(comined_train_test_independent)



comined_train_test_independent


# %%


train_independent['class'].value_counts(), test_independent['class'].value_counts()


# %%


# combine the train and test data and then run pca and split them based on the correct 60% idx point given

def pca(combined_df,split_idx):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    label=combined_df['class'].copy()
    combined_df.drop("class", axis=1, inplace=True)
    
    # Get column names first
    names = combined_df.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(combined_df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    
    
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(scaled_df)


    pca_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1',
                                                                 'principal component 2',
                                                                 'principal component 3',
                                                                 'principal component 4',
                                                                 'principal component 5'])
    
    pca_df['class']=label
    combined_df['class']=label
    

    train_pca=pca_df.iloc[:idx_60_percent_independent,:]
    test_pca=pca_df.iloc[idx_60_percent_independent:,:]
    
    return train_pca, test_pca

    
train_pca_independent, test_pca_independent =pca(comined_train_test_independent,idx_60_percent_independent)
train_pca_dependent, test_pca_dependent =pca(comined_train_test_dependent,idx_60_percent_dependent)


# %%


# merged dataset of all the users, we will use this for finding the optimal hyperparameters for all of all models 
# using sklearn GridsearchCV
def pca_nosplit(combined_df):
    from sklearn import preprocessing
    from sklearn.decomposition import PCA
    
    label=combined_df['class'].copy()
    combined_df.drop("class", axis=1, inplace=True)
    
    # Get column names first
    names = combined_df.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(combined_df)
    scaled_df = pd.DataFrame(scaled_df, columns=names)
    
    
    pca = PCA(n_components=5)
    principalComponents = pca.fit_transform(scaled_df)
    
    pca_df = pd.DataFrame(data = principalComponents, columns = ['principal component 1',
                                                                 'principal component 2',
                                                                 'principal component 3',
                                                                 'principal component 4',
                                                                 'principal component 5'])
    pca_df['class']=label
    combined_df['class']=label
    
    return pca_df
    
merged_dataset=pd.concat(list(merged_aggregated_emg.values()))
reset_index(merged_dataset)

pca_merged_dataset=pca_nosplit(merged_dataset)


# %%





df = shuffle(pca_merged_dataset,random_state=5)
X=df.iloc[:,:-1]
y=df.iloc[:,-1]



def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    kernel= ['linear', 'rbf']
    param_grid = {'C': Cs, 'gamma' : gammas, 'kernel':kernel}
    
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)


    return grid_search.best_params_,grid_search.cv_results_

def decisiontree_param_selection(X, y, nfolds): 
    criterion = ['gini', 'entropy']
    max_depth = [4,6,8,12]
    splitter=['best','random']
    
    param_grid={'criterion':criterion,  
                'max_depth': max_depth, 
                'splitter': splitter }


    grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=nfolds, n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_params_,grid_search.cv_results_

def rf_param_selection(X,y,nfolds):
    param_grid={'criterion': ['gini','entropy'],'max_depth':[4,6,8,10,12],
                'n_estimators': [3,10,100,300,1000] }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=nfolds,n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_params_,grid_search.cv_results_

def nn_param_selection(X, y, nfolds): 
    param_grid= {'solver': ['lbfgs', 'sgd','adam'], 'max_iter': [500,1000,1500], 
                 'alpha': 10.0 ** -np.arange(1, 7), 'hidden_layer_sizes':np.arange(5, 12),
                 'activation': ['logistic', 'tanh','relu']}

    grid_search = GridSearchCV(neural_network.MLPClassifier(), param_grid, cv=nfolds,n_jobs=-1)
    grid_search.fit(X, y)

    return grid_search.best_params_,grid_search.cv_results_




decisiontree_best_perameters, cvres_decision_tree=decisiontree_param_selection(X, y, 5)
svm_best_perameters,cvres_svm=svc_param_selection(X, y, 5)
rf_best_perameters,cvres_svm=rf_param_selection(X, y, 5)


svm_best_perameters, decisiontree_best_perameters, rf_best_perameters


# %%





# %%


train_x_dependent, train_y_dependent = train_pca_dependent.iloc[:,:-1], train_pca_dependent.iloc[:,-1]
test_x_dependent,test_y_dependent =test_pca_dependent.iloc[:,:-1], test_pca_dependent.iloc[:,-1]

train_x_independent, train_y_independent = train_pca_independent.iloc[:,:-1], train_pca_independent.iloc[:,-1]
test_x_independent,test_y_independent =test_pca_independent.iloc[:,:-1], test_pca_independent.iloc[:,-1]


# %%


{'criterion': 'gini', 'max_depth': 12, 'n_estimators': 300}
rf_model_dependent = RandomForestClassifier(criterion='gini', max_depth=12, n_estimators=3000,random_state=10)
rf_model_dependent.fit(train_x_dependent, train_y_dependent)

rf_model_independent = RandomForestClassifier(criterion='gini',max_depth=12, n_estimators=3000,random_state=10)
rf_model_independent.fit(train_x_independent, train_y_independent)


svm_model_dependent=svm.SVC(C= 10, gamma= 1, kernel= 'rbf',random_state=10)
svm_model_dependent.fit(train_x_dependent, train_y_dependent)

svm_model_independent=svm.SVC(C= 10, gamma= 1, kernel= 'rbf',random_state=10)
svm_model_independent.fit(train_x_independent, train_y_independent)

decisiontree_model_dependent=tree.DecisionTreeClassifier(criterion='gini',max_depth=8, splitter='best',random_state=10)
decisiontree_model_dependent.fit(train_x_dependent, train_y_dependent)

decisiontree_model_independent=tree.DecisionTreeClassifier(criterion='gini',max_depth=8, splitter='best',random_state=10)
decisiontree_model_independent.fit(train_x_independent, train_y_independent)


nn_model_dependent=neural_network.MLPClassifier(activation='logistic', alpha=0.01, hidden_layer_sizes=10,
                                                  max_iter=1500,solver= 'lbfgs',random_state=10)
nn_model_dependent.fit(train_x_dependent, train_y_dependent)

nn_model_independent=neural_network.MLPClassifier(activation='logistic', alpha=0.01, hidden_layer_sizes=10,
                                                  max_iter=1500,solver= 'lbfgs',random_state=10)
nn_model_independent.fit(train_x_independent, train_y_independent)


print("RandomForests dependent-split Score: ",rf_model_dependent.score(test_x_dependent, test_y_dependent))
print("RandomForests independent-split Score: ",rf_model_independent.score(test_x_independent, test_y_independent))

print("SVM dependent-split Score: ",svm_model_dependent.score(test_x_dependent, test_y_dependent))
print("SVM independent-split Score: ",svm_model_independent.score(test_x_independent, test_y_independent))

print("DecisionTree dependent-split Score: ",decisiontree_model_dependent.score(test_x_dependent, test_y_dependent))
print("DecisionTree independent-split Score: ",decisiontree_model_independent.score(test_x_independent, test_y_independent))


print("neural network dependent-split Score: ",nn_model_dependent.score(test_x_dependent, test_y_dependent))
print("neural network independent-split Score: ",nn_model_independent.score(test_x_independent, test_y_independent))






# %%


rf_y_trainpred_dependent=rf_model_dependent.predict(train_x_dependent)
rf_y_trainpred_independent=rf_model_independent.predict(train_x_independent)

svm_y_trainpred_dependent=svm_model_dependent.predict(train_x_dependent)
svm_y_trainpred_independent=svm_model_independent.predict(train_x_independent)

df_y_trainpred_dependent=decisiontree_model_dependent.predict(train_x_dependent)
df_y_trainpred_independent=decisiontree_model_independent.predict(train_x_independent)

nn_y_trainpred_dependent=nn_model_dependent.predict(train_x_dependent)
nn_y_trainpred_independent=nn_model_independent.predict(train_x_independent)

rf_y_testpred_dependent=rf_model_dependent.predict(test_x_dependent)
rf_y_testpred_independent=rf_model_independent.predict(test_x_independent)

svm_y_testpred_dependent=svm_model_dependent.predict(test_x_dependent)
svm_y_testpred_independent=svm_model_independent.predict(test_x_independent)

df_y_testpred_dependent=decisiontree_model_dependent.predict(test_x_dependent)
df_y_testpred_independent=decisiontree_model_independent.predict(test_x_independent)

nn_y_testpred_dependent=nn_model_dependent.predict(test_x_dependent)
nn_y_testpred_independent=nn_model_independent.predict(test_x_independent)


print('random forest train dependent')
print(classification_report(train_y_dependent, rf_y_trainpred_dependent, digits=4))
print('random forest test dependent')
print(classification_report(test_y_dependent, rf_y_testpred_dependent, digits=4))
print('random forest train independent')
print(classification_report(train_y_independent, rf_y_trainpred_independent, digits=4))
print('random forest test independent')
print(classification_report(test_y_independent, rf_y_testpred_independent, digits=4))

print('svm train dependent')
print(classification_report(train_y_dependent, svm_y_trainpred_dependent, digits=4))
print('svm test dependent')
print(classification_report(test_y_dependent, svm_y_testpred_dependent, digits=4))
print('svm train independent')
print(classification_report(train_y_independent, svm_y_trainpred_independent, digits=4))
print('svm test independent')
print(classification_report(test_y_independent, svm_y_testpred_independent, digits=4))

print('decision tree train dependent')
print(classification_report(train_y_dependent, df_y_trainpred_dependent, digits=4))
print('decision tree test dependent')
print(classification_report(test_y_dependent, df_y_testpred_dependent, digits=4))
print('decision tree train independent')
print(classification_report(train_y_independent, df_y_trainpred_independent, digits=4))
print('decision tree test independent')
print(classification_report(test_y_independent, df_y_testpred_independent, digits=4))

print('neural network train dependent')
print(classification_report(train_y_dependent, nn_y_trainpred_dependent, digits=4))
print('neural network test dependent')
print(classification_report(test_y_dependent, nn_y_testpred_dependent, digits=4))
print('neural network train independent')
print(classification_report(train_y_independent, nn_y_trainpred_independent, digits=4))
print('neural network test independent')
print(classification_report(test_y_independent, nn_y_testpred_independent, digits=4))








