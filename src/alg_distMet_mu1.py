# coding: utf-8

# In[1]:

import argparse
import sys
import numpy as np
import pandas as pd
from timeit import default_timer as timer
# import pylab as pl
import math, pickle
from scipy import sparse

# In[2]:

parser = argparse.ArgumentParser(description="experiment for mf")
parser.add_argument("C_df_Out_Path", help="path to output altered matrix")
parser.add_argument("userFile_path", help="path to user file")
parser.add_argument("predictions_path", help="path to prediction matrix")
parser.add_argument("topK", help="Top K")
parser.add_argument("epsilon", help="epsilon")

number_of_users_in_testSet = int(6040 * 0.3)

# In[3]:

try:
    args = parser.parse_args()
    runAtCluster = True
    C_df_Out_Path = args.C_df_Out_Path
    userFile_Path = args.userFile_path
    predictionsFile_Path = args.predictions_path
    topK = int(args.topK)
    epsilon = float(args.epsilon)
    topKSquare = topK * topK
    exId = np.random.random()
except:
    print('Please specify arguments correctly.')
    sys.exit()

# In[4]:

alg = 1
logOutputPath = predictionsFile_Path + '_topK_' + str(topK) + '_eps_' + str(
    epsilon) + '_alg1_experimentResults_exId_' + str(exId)

# In[5]:

b_DF = pd.read_csv(userFile_Path)
b_DF.columns = ['userIndex', 'gender']
if 'reddit' in predictionsFile_Path:
    b_DF['gender'] = b_DF['gender'].replace(['male'], 0).replace(['female'], 1)
else:
    b_DF['gender'] = b_DF['gender'].replace(['M'], 0).replace(['F'], 1)

# gender operation
if b_DF['gender'].sum() > float(b_DF['gender'].shape[0]) / 2:
    b_DF['gender'] = b_DF['gender'].replace([0], 2).replace([1], 0).replace([2], 1)
    print('b=0 represents females, b=1 represents males')
else:
    print('b=0 represents males, b=1 represents females')

b_DF.columns = ['userIndex', 'b']

# import predictions
predictionsDF = pd.read_csv(predictionsFile_Path)
predictionsDF.columns = ['userIndex', 'itemIndex', 'position', 'weight']

users_in_C = np.unique(predictionsDF['userIndex'])
number_of_users_in_C = len(users_in_C)

items = list(set(predictionsDF['itemIndex'].tolist()))
number_of_items = len(items)

print('number of users_in_C: ' + str(number_of_users_in_C))
print('number of items: ' + str(number_of_items))

predictionsDF_pruned = predictionsDF[predictionsDF['position'] < topK][['userIndex', 'itemIndex']]

userRecommendationVectors = {}
for userIndex, subFrame in predictionsDF_pruned.groupby('userIndex'):
    userRecommendationVectors[userIndex] = subFrame['itemIndex'].values.tolist()

C_df = pd.DataFrame(userRecommendationVectors).T
C_df = pd.merge(C_df, b_DF, how='left', left_index=True, right_on='userIndex').astype(int)
C_df.index = C_df['userIndex']
del C_df['userIndex']
C_df_original = C_df.copy()

b1 = sum(C_df_original['b'].tolist())
t = len(C_df_original['b'].tolist())
b0 = t - b1


# In[6]:

def measureMetrics(C, userTestRatedItems):
    avPrecision = 0.
    avRecall = 0.

    for userIndex in userTestRatedItems.iterkeys():

        userTestItemsSet = set(userTestRatedItems[userIndex])

        try:
            userRecommendedItemsSet = set(C.loc[userIndex][range(topK)].tolist())
        except:
            userRecommendedItemsSet = set()

        intersection = userRecommendedItemsSet.intersection(userTestItemsSet)
        intersectionSize = len(intersection)

        if intersectionSize == 0:
            precision, recall = 0, 0
        else:

            precision = float(intersectionSize) / len(userRecommendedItemsSet)
            recall = float(intersectionSize) / len(userTestItemsSet)

        avPrecision = avPrecision + precision
        avRecall = avRecall + recall

    avPrecision = avPrecision / len(userTestRatedItems)
    avRecall = avRecall / len(userTestRatedItems)

    return avPrecision, avRecall


'''
def computeUserDistanceForPs3_Old(sourceVector, destinationVector):

    similarity = 0.
    for item_i in sourceVector:
        for item_j in destinationVector:
            if item_i == item_j:
                similarity = similarity + 1
            else:
                try:
                    similarity = similarity + itemSimilarities[str(int(min(item_i, item_j))) + ' ' + str(int(max(item_i,item_j)))]
                except:
                    similarity = similarity

    distance = topKSquare - similarity
    return distance

def computeUserDistanceForPs3(sourceVector, destinationVector):

    sourceVector = str(sorted(list(sourceVector)))
    destinationVector = str(sorted(list(destinationVector)))

    return met3Dict[min(sourceVector,destinationVector) + ' ' + max(sourceVector,destinationVector)]    
'''


def computeObjectiveDistance(C_original, C_modif):
    avDist_1 = 0.
    avDist_2 = 0.
    avDist_3 = 0.

    for userIndex in userRecommendationVectors.keys():
        userOriginalItemsSet = set(C_original.loc[userIndex].tolist())
        userModifItemsSet = set(C_modif.loc[userIndex].tolist())

        dist1 = 1 - (userOriginalItemsSet == userModifItemsSet)
        dist2 = topK - len(userOriginalItemsSet.intersection(userModifItemsSet))
        # dist3 = computeUserDistanceForPs3(userOriginalItemsSet, userModifItemsSet)

        avDist_1 = avDist_1 + dist1
        avDist_2 = avDist_2 + dist2
        # avDist_3 = avDist_3 + dist3

    # return avDist_1, avDist_2, avDist_3
    return avDist_1, avDist_2


def calculateMetricsUserwise(vector, userIndex, C_df, userTestRatedItems):
    userRecommendedItemsSet = set(vector)
    userTestItemsSet = set(userTestRatedItems[userIndex])

    intersection = userRecommendedItemsSet.intersection(userTestItemsSet)
    intersectionSize = len(intersection)

    if intersectionSize == 0:
        precision, recall = 0, 0
    else:
        precision = float(intersectionSize) / len(userRecommendedItemsSet)
        recall = float(intersectionSize) / len(userTestItemsSet)

    return precision, recall


def calculateChangeInDistance(sourceVector, destinationVector, avDist_1, avDist_2):
    userOriginalItemsSet = set(sourceVector)
    userModifItemsSet = set(destinationVector)

    dist1 = 1 - (userOriginalItemsSet == userModifItemsSet)
    dist2 = topK - len(userOriginalItemsSet.intersection(userModifItemsSet))
    # dist3 = computeUserDistanceForPs3(userOriginalItemsSet, userModifItemsSet)

    avDist_1 = avDist_1 + dist1
    avDist_2 = avDist_2 + dist2
    # avDist_3 = avDist_3 + dist3

    # return avDist_1, avDist_2, avDist_3
    return avDist_1, avDist_2


def calculateChangeInMetrics(sourceVector, destinationVector, userIndex, C_df, userTestRatedItems, precision, recall):
    precisionOfExVector, recallOfExVector = calculateMetricsUserwise(sourceVector, userIndex, C_df, userTestRatedItems)
    precisionOfNewVector, recallOfNewVector = calculateMetricsUserwise(destinationVector, userIndex, C_df,
                                                                       userTestRatedItems)

    newPrecision = ((
                                precision * number_of_users_in_testSet) - precisionOfExVector + precisionOfNewVector) / number_of_users_in_testSet
    newRecall = ((
                             recall * number_of_users_in_testSet) - recallOfExVector + recallOfNewVector) / number_of_users_in_testSet

    return newPrecision, newRecall


# In[7]:

def compute_BER(summaryDF):
    BER_firstComponent = summaryDF.groupby('Y_x')['H(0,y)'].sum()[1] / float(b0)
    BER_secondComponent = summaryDF.groupby('Y_x')['H(1,y)'].sum()[0] / float(b1)
    sigma = 0.5 * (BER_firstComponent + BER_secondComponent)
    return sigma


def computeSummaryDF(C_df):
    # input = C_df, b
    # output = C_df >> H(1,y), H(0,y), s(y) for all y \in Y and Y1, Y0
    grouper = np.arange(topK).tolist()
    summaryDF = C_df.groupby(grouper).agg([np.sum, np.size])
    summaryDF.columns = ['H(1,y)', 'size']
    summaryDF = summaryDF.reset_index()
    summaryDF['H(0,y)'] = summaryDF['size'] - summaryDF['H(1,y)']
    summaryDF['s(y)'] = (summaryDF['H(0,y)'] / float(b0)) - (summaryDF['H(1,y)'] / float(b1))
    summaryDF['Y_x'] = np.sign(summaryDF['s(y)']).replace([1], 0).replace([-1], 1)

    return summaryDF


def cureWoman(Y_x, H_0_y, H_1_y, q_y):
    if Y_x == 1 and H_0_y == 0:
        return H_1_y
    else:
        return q_y


def compute_BER_d_p_q_k_l_beforeMovingWoman(summaryDF):
    # input = summaryDF
    # output = sigma, d, k1, k2, k, _2_b1d, l

    BER_firstComponent = summaryDF.groupby('Y_x')['H(0,y)'].sum()[1] / float(b0)
    BER_secondComponent = summaryDF.groupby('Y_x')['H(1,y)'].sum()[0] / float(b1)
    sigma = 0.5 * (BER_firstComponent + BER_secondComponent)

    d = epsilon - sigma
    if d <= 0:
        print('C is fair enough <3 <3 <3')

    summaryDF['p(y)'] = (summaryDF['s(y)'] * b1).apply(math.floor)
    summaryDF['q(y)'] = (summaryDF['s(y)'] * (-1 * b1) - 1).apply(math.ceil)
    summaryDF['q(y)'] = np.vectorize(cureWoman)(summaryDF['Y_x'], summaryDF['H(0,y)'], summaryDF['H(1,y)'],
                                                summaryDF['q(y)'])

    k1 = summaryDF.groupby('Y_x')['p(y)'].sum()[0]
    k2 = summaryDF.groupby('Y_x')['q(y)'].sum()[1]
    k = min(k1, k2)

    _2_b1d = math.ceil(2. * b1 * d)
    l = int(min(_2_b1d, k))
    math.ceil(2. * b1 * d), k, l

    return sigma, d, k1, k2, k, _2_b1d, l


def vectorizeCartesian(Y_0, Y_1):
    return str(Y_0) + ' ' + str(Y_1)


def compute_BER_d_p_q_k_l_beforeMovingMan(summaryDF):
    # input = summaryDF
    # output = sigma, d, k1, k2, k, _2_b1d, l

    BER_firstComponent = summaryDF.groupby('Y_x')['H(0,y)'].sum()[1] / float(b0)
    BER_secondComponent = summaryDF.groupby('Y_x')['H(1,y)'].sum()[0] / float(b1)
    sigma = 0.5 * (BER_firstComponent + BER_secondComponent)

    d = epsilon - sigma
    if d <= 0:
        print('C is fair enough <3 <3 <3')

    summaryDF['p(y)'] = (summaryDF['s(y)'] * b0).apply(math.floor)
    summaryDF['q(y)'] = (summaryDF['s(y)'] * (-1 * b0) - 1).apply(math.ceil)

    k1 = summaryDF.groupby('Y_x')['p(y)'].sum()[0]
    k2 = summaryDF.groupby('Y_x')['q(y)'].sum()[1]
    k = min(k1, k2)

    _2_b0d = math.ceil(2. * b0 * d)
    l = int(min(_2_b0d, k))
    math.ceil(2. * b0 * d), k, l

    return sigma, d, k1, k2, k, _2_b0d, l


# In[8]:

summaryDF = computeSummaryDF(C_df)
sigma, d, k1, k2, k, _2_b1d, l = compute_BER_d_p_q_k_l_beforeMovingWoman(summaryDF)
initialBer, numberOfWomansToBeMoved = sigma, l

print('initialBer = ' + str(initialBer))
print('numberOfWomansToBeMoved = ' + str(numberOfWomansToBeMoved))

# In[9]:

if d <= 0:
    print('C is fair')
    C_df.to_csv(C_df_Out_Path)
    sys.exit()

# In[10]:

# lists of metrics and initial computations

precisionList = []
recallList = []
dist1List = []
dist2List = []
dist3List = []
BERList = []
numberOfPeopleMoved = 0
numberOfPeopleMovedList = []
deltaBerList = []

# avPrecision, avRecall = measureMetrics(C_df, userTestRatedItems)
ber = initialBer
# avDist1, avDist2, avDist3 = computeObjectiveDistance(C_df_original[range(topK)], C_df[range(topK)])
avDist1, avDist2 = computeObjectiveDistance(C_df_original[range(topK)], C_df[range(topK)])

# precisionList.append(avPrecision)
# recallList.append(avRecall)
dist1List.append(avDist1)
dist2List.append(avDist2)
# dist3List.append(avDist3)
BERList.append(ber)
numberOfPeopleMovedList.append(numberOfPeopleMoved)

# Move $\ell$ member of b1 (need to elaborate on the meaning of ``move'').
#
# Meaning of move: for a single move, do it randomly

# In[11]:

# algorithm1
# move l b1 operation
movingClockStart = timer()
for i in range(int(l)):
    sourceVectorRow = summaryDF[summaryDF['q(y)'] > 0].sample()
    sourceVector = sourceVectorRow.values[0][:topK].astype(int)
    sourceVectorIndex = sourceVectorRow.index[0]

    destinationVectorRow = summaryDF[summaryDF['p(y)'] > 0].sample()
    destinationVector = destinationVectorRow.values[0][:topK].astype(int)
    destinationVectorIndex = destinationVectorRow.index[0]

    womanSubjectToMoveDf = C_df.loc[C_df[C_df[range(topK)] == sourceVector].dropna(thresh=topK).index]
    womanSubjectToMoveIndex = womanSubjectToMoveDf[womanSubjectToMoveDf['b'] == 1].sample().index[0]

    C_df.loc[womanSubjectToMoveIndex, range(topK)] = destinationVector

    summaryDF.loc[sourceVectorIndex, 'H(1,y)'] = summaryDF.loc[sourceVectorIndex, 'H(1,y)'] - 1
    summaryDF.loc[sourceVectorIndex, 'size'] = summaryDF.loc[sourceVectorIndex, 'size'] - 1
    summaryDF.loc[sourceVectorIndex, 's(y)'] = summaryDF.loc[sourceVectorIndex, 's(y)'] + (1. / b1)
    summaryDF.loc[sourceVectorIndex, 'q(y)'] = summaryDF.loc[sourceVectorIndex, 'q(y)'] - 1

    summaryDF.loc[destinationVectorIndex, 'H(1,y)'] = summaryDF.loc[destinationVectorIndex, 'H(1,y)'] + 1
    summaryDF.loc[destinationVectorIndex, 'size'] = summaryDF.loc[destinationVectorIndex, 'size'] + 1
    summaryDF.loc[destinationVectorIndex, 's(y)'] = summaryDF.loc[destinationVectorIndex, 's(y)'] - (1. / b1)
    summaryDF.loc[destinationVectorIndex, 'p(y)'] = summaryDF.loc[destinationVectorIndex, 'p(y)'] - 1

    numberOfPeopleMoved = numberOfPeopleMoved + 1

    # if womanSubjectToMoveIndex in userTestRatedItems.keys():
    #    avPrecision, avRecall = calculateChangeInMetrics(sourceVector, destinationVector, womanSubjectToMoveIndex, C_df, userTestRatedItems, precisionList[-1], recallList[-1])
    # else:
    #    avPrecision, avRecall = precisionList[-1], recallList[-1]

    ber = BERList[-1] + (1. / (2 * b1))
    # avDist1, avDist2, avDist3 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1], dist3List[-1])
    avDist1, avDist2 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1])

    deltaBerList.append(1. / (2 * b1))
    # precisionList.append(avPrecision)
    # recallList.append(avRecall)
    dist1List.append(avDist1)
    dist2List.append(avDist2)
    # dist3List.append(avDist3)
    BERList.append(ber)
    numberOfPeopleMovedList.append(numberOfPeopleMoved)

movingClockEnd = timer()
print('Moving Woman Timer: ' + str(movingClockEnd - movingClockStart))

# In[12]:

summaryDF = computeSummaryDF(C_df)

# In[13]:

sigma_w, d, k1, k2, k, _2_b0d, l = compute_BER_d_p_q_k_l_beforeMovingMan(summaryDF)

d = epsilon - sigma_w
if d <= 0:
    print('C is fair')
    C_df.to_csv(C_df_Out_Path)
    sys.exit()

# In[14]:

sigma_w, d, k1, k2, k, _2_b0d, l
berAfterMovingWoman = sigma_w
numberOfMansToBeMoved = l

print('berAfterMovingWoman = ' + str(berAfterMovingWoman))
print('BER double check = ' + str(BERList[-1]))
print('numberOfMansToBeMoved = ' + str(numberOfMansToBeMoved))

# In[15]:

# algorithm1
# move l b1 operation
movingClockStart = timer()

for i in range(int(l)):
    sourceVectorRow = summaryDF[summaryDF['p(y)'] > 0].sample()
    sourceVector = sourceVectorRow.values[0][:topK].astype(int)
    sourceVectorIndex = sourceVectorRow.index[0]

    destinationVectorRow = summaryDF[summaryDF['q(y)'] > 0].sample()
    destinationVector = destinationVectorRow.values[0][:topK].astype(int)
    destinationVectorIndex = destinationVectorRow.index[0]

    manSubjectToMoveDf = C_df.loc[C_df[C_df[range(topK)] == sourceVector].dropna(thresh=topK).index]
    manSubjectToMoveIndex = manSubjectToMoveDf[manSubjectToMoveDf['b'] == 0].sample().index[0]

    C_df.loc[manSubjectToMoveIndex, range(topK)] = destinationVector

    summaryDF.loc[sourceVectorIndex, 'H(0,y)'] = summaryDF.loc[sourceVectorIndex, 'H(0,y)'] - 1
    summaryDF.loc[sourceVectorIndex, 'size'] = summaryDF.loc[sourceVectorIndex, 'size'] - 1
    summaryDF.loc[sourceVectorIndex, 's(y)'] = summaryDF.loc[sourceVectorIndex, 's(y)'] + (1. / b0)
    summaryDF.loc[sourceVectorIndex, 'p(y)'] = summaryDF.loc[sourceVectorIndex, 'p(y)'] - 1

    summaryDF.loc[destinationVectorIndex, 'H(0,y)'] = summaryDF.loc[destinationVectorIndex, 'H(0,y)'] + 1
    summaryDF.loc[destinationVectorIndex, 'size'] = summaryDF.loc[destinationVectorIndex, 'size'] + 1
    summaryDF.loc[destinationVectorIndex, 's(y)'] = summaryDF.loc[destinationVectorIndex, 's(y)'] - (1. / b0)
    summaryDF.loc[destinationVectorIndex, 'q(y)'] = summaryDF.loc[destinationVectorIndex, 'q(y)'] - 1

    numberOfPeopleMoved = numberOfPeopleMoved + 1

    # if manSubjectToMoveIndex in userTestRatedItems.keys():
    #    avPrecision, avRecall = calculateChangeInMetrics(sourceVector, destinationVector, manSubjectToMoveIndex, C_df, userTestRatedItems, precisionList[-1], recallList[-1])
    # else:
    #    avPrecision, avRecall = precisionList[-1], recallList[-1]

    ber = BERList[-1] + (1. / (2 * b0))
    # avDist1, avDist2, avDist3 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1], dist3List[-1])
    avDist1, avDist2 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1])

    deltaBerList.append(1. / (2 * b0))
    # precisionList.append(avPrecision)
    # recallList.append(avRecall)
    dist1List.append(avDist1)
    dist2List.append(avDist2)
    # dist3List.append(avDist3)
    BERList.append(ber)
    numberOfPeopleMovedList.append(numberOfPeopleMoved)

movingClockEnd = timer()
pS = "Moving Man Timer: " + str(movingClockEnd - movingClockStart)
print(pS)

# In[16]:

sigma_wm = compute_BER(summaryDF)
finalBer = sigma_wm
print('finalBer = ' + str(finalBer))
print('BER double check = ' + str(BERList[-1]))

# In[17]:

if epsilon > sigma_wm:
    print('C is still not fair enough')
    print(sigma_wm)

else:
    print('C is fair enough')
    print(sigma_wm)

# In[ ]:

C_df.to_csv(C_df_Out_Path)
