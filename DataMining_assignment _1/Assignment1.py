# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
# %matplotlib inline  


# ran from the root directory of the project folder from the zip file provided 
import os;
cwd = os.getcwd()
print(cwd)

# +
#find the files for all users for groundTruth and the myoData that will be read into pandas via pd.read_csv()

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
       
print('This is emg folders')
print('\n')
print(emg)
print('\n')
print('This is imu folders')
print('\n')
print(imu)
print('\n')
print('this is groundTruth')
print('\n')
print(groundTruth)
 
#since there are 30 users with both fork and spoon data we want to see each of our lists contain 60 samples. 
print(len(groundTruth))        
print(len(emg))
print(len(imu))

     

    


# +
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

print(len(matches))
print(matches)
#confirmed that there are 60 matches for the 30 users grundtruth and myo data from both spoon and fork
# had to change user 09 in myo data to user9 so that we could get an idx match
# have the idx of where the corresponding files from groundtruth and myo data match from our list of files
# confirmed that imu_matches is equal to emg_matches


# +
user_idx_imu=[]
imu_match=[]
for i in range(len(groundTruth)):
 
    user_idx_imu.append(emg[i].split('/')[-3:-1])
for i in range(len(user_idx_imu)):
    if user_idx_imu[i][0]=='user09':
        user_idx_imu[i][0]='user9'
for i in range(len(user_idx_groundTruth)):

    for j in range (len(user_idx_imu)):
         
        if user_idx_groundTruth[i]==user_idx_imu[j]:
            imu_match.append((user_idx_groundTruth[i][0],user_idx_groundTruth[i][1],i,j))
            
imu_match==matches
            


# +

#matches tupes are in the format('user_id',spoon/fork,groundtruth_file_idx,myo_file_idx)
# these are going to be used so that we can create a dict[userid_utensiltype]=groundtruth_df
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


# since we are using python we want the row numbers to be (797,930)->(796,930) which we have achieved 
# I dont believe that changing the sample by 1 row will effect our seperation regardless. by not changing the 
# stop idx by 1 we have still achieved the same thing since our idx slices will only go up to 929
groundTruth_dict['user9_fork'].head(1)


# -

def CreateDict(file_list,matches,col_names):
    user_dict={}
    for item in matches:
        user_id,user_utensil,groundTruth_idx,myo_idx =item
        df_name=user_id+'_'+user_utensil
        df=pd.read_csv(file_list[myo_idx],names=col_names)
        user_dict[df_name]=df
    return user_dict



# +
#create a dict[userid_utensiltype]=groundtruth_df

emg_col_names=['timestamp','EMG1','EMG2','EMG3','EMG4','EMG5','EMG6','EMG7','EMG8']
imu_col_names=['timestamp', 'Orientation X', 'Orientation Y', 'Orientation Z', 
               'Orientation W', 'Accelerometer X', 'Accelerometer Y', 'Accelerometer Z', 
               'Gyroscope X', 'Gyroscope Y','Gyroscope Z']

emg_dict=CreateDict(emg,matches,emg_col_names)
imu_dict=CreateDict(imu,matches,imu_col_names)




# +
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




# +

def getAllActions(groundTruth_dict,myo_dict,col_names,random_sampling=True):
    eating=pd.DataFrame(columns=col_names)
    non_eating=pd.DataFrame(columns=col_names)
    user_action_dict={}
    for user_id,df in groundTruth_dict.items():

    
        eat_user,not_eat_user=findActionsUser(groundTruth_dict[user_id],myo_dict[user_id],col_names)
        #print(user_id)
        #create a dict with individual user actions 

        
        eat_user['class']='eat'
        not_eat_user['class']='not_eat'


        eating=eating.append(eat_user, sort=True)
        if random_sampling:
            non_eating=non_eating.append(not_eat_user.sample(len(eat_user),random_state=10),sort=True)
            user_action_dict[user_id]=[eat_user,not_eat_user.sample(len(eat_user),random_state=10)]
            
            
        else:
            non_eating=non_eating.append(not_eat_user)
            user_action_dict[user_id]=[eat_user,not_eat_user]
           
                
    return eating,non_eating,user_action_dict


eat_emg,not_eat_emg,emg_action_dict=getAllActions(groundTruth_dict,emg_dict,emg_col_names)
eat_imu,not_eat_imu,imu_action_dict=getAllActions(groundTruth_dict,imu_dict,imu_col_names)




# +
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
    
  
    
# -

fig,axes=plt.subplots(nrows=2,figsize=(12,12))
axes[0].scatter(not_eat_emg.index,not_eat_emg.EMG8,s=1, color='r')
axes[1].scatter(eat_emg.index,eat_emg.EMG8,s=1, color='k')

axes[0].set(xlabel="index",ylabel='EMG8')
axes[1].set(xlabel="index",ylabel='EMG8')
axes[0].set_title('Non-Eating ')
axes[1].set_title('Eating ')
plt.show()
plt.clf()


# +

def eat_not_eat_emg_statistics(eat,not_eat):
    
    not_eat_feat=not_eat.iloc[:,1:-2]
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
    
    eat_feat=eat.iloc[:,1:-2]
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
    
    return eat_statistics, not_eat_statistics

emg_eat_feature_matrix,emg_not_eat_feature_matrix=eat_not_eat_emg_statistics(eat_emg,not_eat_emg)
imu_eat_feature_matrix,imu_not_eat_feature_matrix=eat_not_eat_emg_statistics(eat_imu,not_eat_imu)







# +
# creating dictionarys containing each users statistical_feature_matrix of [eating,non_eating]

emg_action_dict_feature_matrix={}
imu_action_dict_feature_matrix={}

for k in emg_action_dict.keys():
    emg_eat,emg_no_eat=eat_not_eat_emg_statistics(emg_action_dict[k][0],emg_action_dict[k][1])
    emg_action_dict_feature_matrix[k]=[emg_eat,emg_no_eat]
    
    imu_eat,imu_no_eat=eat_not_eat_emg_statistics(imu_action_dict[k][0],imu_action_dict[k][1])
    imu_action_dict_feature_matrix[k]=[imu_eat,imu_no_eat]


# +
# plotting all of the statistical features against their index for eating and non_eating 
for col in range(emg_eat_feature_matrix.shape[1]):
    fig,axes=plt.subplots(nrows=2,figsize=(12,12))
    axes[0].scatter(emg_not_eat_feature_matrix.index,emg_not_eat_feature_matrix.iloc[:,col],s=1, color='r')
    axes[1].scatter(emg_eat_feature_matrix.index,emg_eat_feature_matrix.iloc[:,col],s=1, color='k')
    
    axes[0].set(xlabel="index",ylabel=emg_eat_feature_matrix.columns[col])
    axes[1].set(xlabel="index",ylabel=emg_eat_feature_matrix.columns[col])
    

    #axes[0].legend([' NOT- EATING'])
    #axes[1].legend([' EATING'])
    axes[0].set_title('Non-Eating '+ emg_eat_feature_matrix.columns[col]+' Row-wise')
    axes[1].set_title('Eating '+ emg_eat_feature_matrix.columns[col]+' Row-wise')

    plt.show()
    plt.clf()
    
print('made')    

# -

for col in range(imu_eat_feature_matrix.shape[1]):
    fig,axes=plt.subplots(nrows=2,figsize=(12,12))
    axes[0].scatter(imu_not_eat_feature_matrix.index,imu_not_eat_feature_matrix.iloc[:,col],s=1, color='r')
    axes[1].scatter(imu_eat_feature_matrix.index,imu_eat_feature_matrix.iloc[:,col],s=1, color='k')
    
    axes[0].set(xlabel="index",ylabel=emg_eat_feature_matrix.columns[col])
    axes[1].set(xlabel="index",ylabel=emg_eat_feature_matrix.columns[col])
    

    #axes[0].legend([' NOT- EATING'])
    #axes[1].legend([' EATING'])
    axes[0].set_title('Non-Eating '+ imu_not_eat_feature_matrix.columns[col]+' Row-wise IMU')
    axes[1].set_title('Eating '+ imu_eat_feature_matrix.columns[col]+' Row-wise IMU')

    plt.show()
    plt.clf()

# +
# create a matrix containing the labels to be added onto after pca. and create the combined
# emg matrix without class labels for pca 

emg_eat_feature_matrix['class']='eat'
emg_not_eat_feature_matrix['class']='not eat'

labels_eat=pd.DataFrame()
labels_not_eat=pd.DataFrame()

labels_not_eat['class']=emg_not_eat_feature_matrix.loc[:,'class']
labels_eat['class']=emg_eat_feature_matrix.loc[:,'class']

labels=[labels_eat,labels_not_eat]
all_labels = pd.concat(labels)
reset_index(all_labels)

eat_pca=emg_eat_feature_matrix.drop(['class'], axis=1)
no_eat_pca=emg_not_eat_feature_matrix.drop(['class'], axis=1)

combined=[eat_pca,no_eat_pca]
pca_matrix = pd.concat(combined)
reset_index(pca_matrix)
pca_matrix








# -

# we need to standardize our data since they are all on different scales. 
# that way the data of larger scale isn't contributing to most of the variance
# in pca. Standardizing the data subtracts by the mean and divides by the standard deviation 
# we can use sklearns preprocessing to do this 


from sklearn import preprocessing
# Get column names first
names = pca_matrix.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit your data on the scaler object
scaled_df = scaler.fit_transform(pca_matrix)
scaled_df = pd.DataFrame(scaled_df, columns=names)

# +
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(scaled_df)


pca_df = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2',
                         'principal component 3','principal component 4','principal component 5'])
# -

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('Variance Explained %');
plt.title('Principle components explained Variance %')
names=['pc1','pc2','pc3','pc4','pc5',]
l=list(range(len(pca.explained_variance_ratio_)))
plt.bar(names,pca.explained_variance_ratio_)

# +
plt.figure(figsize=(12,12))
halfway_idx=int(630268/2)
pca_df_eat=pca_df.iloc[:halfway_idx,:]
pca_df_not_eat=pca_df.iloc[halfway_idx:,:]



fig,axes=plt.subplots(nrows=2,figsize=(12,12))
axes[1].scatter(pca_df_eat.index,pca_df_eat['principal component 1'],s=1, color='k')
axes[0].scatter(pca_df_not_eat.index,pca_df_not_eat['principal component 1'],s=1, color='r')

axes[0].set(xlabel="index",ylabel='PC 1')
axes[1].set(xlabel="index",ylabel='PC 1')


#axes[0].legend([' NOT- EATING'])
#axes[1].legend([' EATING'])
axes[1].set_title('Eating ')
axes[0].set_title('Non-Eating ')

plt.show()
plt.clf()

# +
plt.figure(figsize=(12,12))
halfway_idx=int(630268/2)
pca_df_eat=pca_df.iloc[:halfway_idx,:]
pca_df_not_eat=pca_df.iloc[halfway_idx:,:]



fig,axes=plt.subplots(nrows=2,figsize=(12,12))
axes[1].scatter(pca_df_eat.index,pca_df_eat['principal component 2'],s=1, color='k')
axes[0].scatter(pca_df_not_eat.index,pca_df_not_eat['principal component 2'],s=1, color='r')

axes[0].set(xlabel="index",ylabel='PC 2')
axes[1].set(xlabel="index",ylabel='PC 2')


#axes[0].legend([' NOT- EATING'])
#axes[1].legend([' EATING'])
axes[1].set_title('Eating ')
axes[0].set_title('Non-Eating ')

plt.show()
plt.clf()
# -










