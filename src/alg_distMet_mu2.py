
# coding: utf-8

# In[1]:

import argparse
import sys
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import sys
import math, pickle
from scipy import sparse



parser = argparse.ArgumentParser( description = "experiment for mf" )
parser.add_argument( "C_df_Out_Path", help = "path to output altered matrix" )
parser.add_argument( "userFile_path", help = "path to user file" )
parser.add_argument( "predictions_path", help = "path to prediction matrix" )
parser.add_argument( "topK", help = "Top K" )
parser.add_argument( "epsilon", help = "epsilon" )
# In[2]:

try:
    args = parser.parse_args()
    runAtCluster = True
    C_df_Out_Path = args.C_df_Out_Path
    userFile_Path = args.userFile_path
    predictionsFile_Path = args.predictions_path
    topK = int(args.topK)
    epsilon = float(args.epsilon)
    topKSquare = topK*topK
    exId = np.random.random()
    
except:
    print 'Please specify parameters correctly'
    sys.exit()


# In[3]:

alg = 2
logOutputPath = predictionsFile_Path + '_topK_' + str(topK) + '_eps_' + str(epsilon) + '_alg1_experimentResults_exId_' + str(exId)


# In[ ]:




# In[4]:

b_DF = pd.read_csv(userFile_Path)
b_DF.columns = ['userIndex','gender']
if 'reddit' in predictionsFile_Path:
    b_DF['gender'] = b_DF['gender'].replace(['male'], 0).replace(['female'], 1)
else:
    b_DF['gender'] = b_DF['gender'].replace(['M'], 0).replace(['F'], 1)

#gender operation
if b_DF['gender'].sum() > float(b_DF['gender'].shape[0])/2:
    b_DF['gender'] = b_DF['gender'].replace([0], 2).replace([1], 0).replace([2], 1)
    print 'b=0 represents females, b=1 represents males'
else:
    print 'b=0 represents males, b=1 represents females'
    
b_DF.columns = ['userIndex','b']
    
# import predictions
predictionsDF = pd.read_csv(predictionsFile_Path)
predictionsDF.columns = ['userIndex', 'itemIndex', 'position', 'weight']

users_in_C = np.unique(predictionsDF['userIndex'])
number_of_users_in_C = len(users_in_C)


items = list(set(predictionsDF['itemIndex'].tolist()))
number_of_items = len(items)

print 'number of users_in_C: ' + str(number_of_users_in_C) 
print 'number of items: ' + str(number_of_items)

predictionsDF_pruned = predictionsDF[predictionsDF['position'] < topK][['userIndex','itemIndex']]

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


# In[5]:

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
            precision, recall = 0 ,0
        else:
            
            precision = float(intersectionSize) / len(userRecommendedItemsSet)
            recall = float(intersectionSize) / len(userTestItemsSet)

        avPrecision = avPrecision + precision
        avRecall = avRecall + recall
        

    avPrecision = avPrecision / len(userTestRatedItems)
    avRecall = avRecall / len(userTestRatedItems)

    return avPrecision, avRecall

def computeUserDistanceForPs3_Old(sourceVector, destinationVector):

    similarity = 0.
    for item_i in sourceVector:
        for item_j in destinationVector:
            if item_i == item_j:
                similarity = similarity + 1
            else:
                try:
                    similarity = similarity + itemSimilarities[str(int(min(item_i,item_j))) + ' ' + str(int(max(item_i,item_j)))]
                except:
                    similarity = similarity
     
    distance = topKSquare - similarity
    return distance

def computeUserDistanceForPs3(sourceVector, destinationVector):
    
    sourceVector = str(sorted(list(sourceVector)))
    destinationVector = str(sorted(list(destinationVector)))
    
    return met3Dict[min(sourceVector,destinationVector) + ' ' + max(sourceVector,destinationVector)]    


def computeObjectiveDistance(C_original, C_modif):
        
    avDist_1 = 0.
    avDist_2 = 0.
    avDist_3 = 0.
    
    for userIndex in userRecommendationVectors.iterkeys():
        
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
        precision, recall = 0,0
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
    precisionOfNewVector, recallOfNewVector = calculateMetricsUserwise(destinationVector, userIndex, C_df, userTestRatedItems)
    
    newPrecision = ((precision*number_of_users_in_testSet) - precisionOfExVector + precisionOfNewVector)/number_of_users_in_testSet
    newRecall = ((recall*number_of_users_in_testSet) - recallOfExVector + recallOfNewVector)/number_of_users_in_testSet
    
    return newPrecision, newRecall
    


# In[6]:

def compute_BER(summaryDF):
    BER_firstComponent = summaryDF.groupby('Y_x')['H(0,y)'].sum()[1] / float(b0)
    BER_secondComponent = summaryDF.groupby('Y_x')['H(1,y)'].sum()[0] / float(b1)
    sigma = 0.5 * (BER_firstComponent + BER_secondComponent)
    return sigma

def computeSummaryDF(C_df):
    # input = C_df, b
    # output = C_df >> H(1,y), H(0,y), s(y) for all y \in Y and Y1, Y0
    
    summaryDF = C_df.groupby(range(topK)).agg([np.sum, np.size])
    summaryDF.columns = ['H(1,y)', 'size']
    summaryDF = summaryDF.reset_index()
    summaryDF['H(0,y)'] = summaryDF['size'] - summaryDF['H(1,y)']
    summaryDF['s(y)'] = (summaryDF['H(0,y)']/float(b0)) - (summaryDF['H(1,y)']/float(b1))
    summaryDF['Y_x'] = np.sign(summaryDF['s(y)']).replace([1], 0).replace([-1], 1)
    
    return summaryDF

def convertUserRecommendationVectorsToPredictionMatrix(userRecommendationVectors, number_of_items):
    
    predictionMatrix = np.zeros((number_of_users, number_of_items),dtype=int)
    
    userIndex = 0
    for userVector in userRecommendationVectors:
        for item in userVector:
            predictionMatrix[userIndex, int(item)] = 1
        userIndex = userIndex + 1
            
    return predictionMatrix   

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
        print 'C is fair enough <3 <3 <3'
       
    summaryDF['p(y)'] = (summaryDF['s(y)'] * b1).apply(math.floor)
    summaryDF['q(y)'] = (summaryDF['s(y)'] * (-1*b1) -1).apply(math.ceil)
    summaryDF['q(y)'] = np.vectorize(cureWoman)(summaryDF['Y_x'], summaryDF['H(0,y)'], summaryDF['H(1,y)'], summaryDF['q(y)'])
    
    k1 = summaryDF.groupby('Y_x')['p(y)'].sum()[0]
    k2 = summaryDF.groupby('Y_x')['q(y)'].sum()[1]
    k = min(k1,k2)

    _2_b1d = math.ceil(2.*b1*d)
    l = int(min(_2_b1d, k))
    math.ceil(2.*b1*d), k, l
    
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
        print 'C is fair enough <3 <3 <3'
       
    summaryDF['p(y)'] = (summaryDF['s(y)'] * b0).apply(math.floor)
    summaryDF['q(y)'] = (summaryDF['s(y)'] * (-1*b0) -1).apply(math.ceil)
    
    k1 = summaryDF.groupby('Y_x')['p(y)'].sum()[0]
    k2 = summaryDF.groupby('Y_x')['q(y)'].sum()[1]
    k = min(k1,k2)

    _2_b0d = math.ceil(2.*b0*d)
    l = int(min(_2_b0d, k))
    math.ceil(2.*b0*d), k, l
    
    return sigma, d, k1, k2, k, _2_b0d, l


# In[7]:

summaryDF = computeSummaryDF(C_df)
sigma, d, k1, k2, k, _2_b1d, l = compute_BER_d_p_q_k_l_beforeMovingWoman(summaryDF)
initialBer, numberOfWomansToBeMoved = sigma, l

print 'initialBer = ' + str(initialBer)
print 'numberOfWomansToBeMoved = ' + str(numberOfWomansToBeMoved)


# In[8]:

if d <= 0:
    print 'C is fair'
    C_df.to_csv(C_df_Out_Path)
    sys.exit()


# In[9]:

if l > 0:
    
    # compute V1 and V0
    movementPossibleY_1 = summaryDF[(summaryDF['Y_x'] == 1) & (summaryDF['q(y)'] > 0)]
    movementPossibleY_0 = summaryDF[(summaryDF['Y_x'] == 0) & (summaryDF['p(y)'] > 0)]

    V1 = []
    for i in movementPossibleY_1.index:
        q_z = int(movementPossibleY_1.loc[i, 'q(y)'])
        for q_z_index in range(q_z):
            V1.append(str(int(i))+'_'+ str(q_z_index))

    V0 = []
    for i in movementPossibleY_0.index:
        p_y = int(movementPossibleY_0.loc[i, 'p(y)'])
        for p_y_index in range(p_y):
            V0.append(str(int(i))+'_'+ str(p_y_index))

    print '|V0| = ' + str(len(V0))
    print '|V1| = ' + str(len(V1))


    # compute summaryDF dictionary version for computational reasons
    summaryDF_ForCartesian = summaryDF[range(topK)]
    summaryDF_ForCartesian_Dict = summaryDF_ForCartesian.T.to_dict()
    for key in summaryDF_ForCartesian_Dict.iterkeys():
        summaryDF_ForCartesian_Dict[key] = set(summaryDF_ForCartesian_Dict[key].values())
   
    flowStart = timer()
    V_dictionary_orj_to_target = {}
    V_dictionary_target_to_orj = {}

    c=2
    for i in V0:
        V_dictionary_orj_to_target[i] = c
        V_dictionary_target_to_orj[c] = i
        c = c +1

    for i in V1:
        V_dictionary_orj_to_target[i] = c
        V_dictionary_target_to_orj[c] = i
        c = c +1


    from ortools.graph import pywrapgraph

    v_s = 0
    v_t = 1

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []

    supplies = [l,-l]

    for i_v0 in V0:
        start_nodes.append(v_s)
        end_nodes.append(V_dictionary_orj_to_target[i_v0])
        capacities.append(1)
        unit_costs.append(0)

    for i_v1 in V1:
        start_nodes.append(V_dictionary_orj_to_target[i_v1])
        end_nodes.append(v_t)
        capacities.append(1)
        unit_costs.append(0)

    for i_v0 in V0:
        y0i = i_v0.split('_')[0]
        for i_v1 in V1:
            y1i = i_v1.split('_')[0]

            start_nodes.append(V_dictionary_orj_to_target[i_v0])
            end_nodes.append(V_dictionary_orj_to_target[i_v1])
            capacities.append(1)
            cost = len(summaryDF_ForCartesian_Dict[int(y0i)] - summaryDF_ForCartesian_Dict[int(y1i)])*2
            unit_costs.append(cost)

    for i in range(len(V0+V1)):
        supplies.append(0)


    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],capacities[i], unit_costs[i])

    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
           
    resultList = []
    # Find the minimum cost flow between node 0 and node 4.
    womanToBeMovedTable = []
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        min_cost_flow.Solve()
        print('Minimum cost:', min_cost_flow.OptimalCost())
        for i in range(min_cost_flow.NumArcs()):
            if (min_cost_flow.Flow(i) != 0) and (min_cost_flow.Tail(i) != 0) and (min_cost_flow.Head(i) != 1):
                womanToBeMovedTable.append([V_dictionary_target_to_orj[min_cost_flow.Tail(i)],V_dictionary_target_to_orj[min_cost_flow.Head(i)]])
    else:
        print('There was an issue with the min cost flow input.')

    flowEnds = timer()

    print "Min cost flow algorithm calculation time (seconds): " + str(flowEnds - flowStart)


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

#avPrecision, avRecall = measureMetrics(C_df, userTestRatedItems)
ber = initialBer
# avDist1, avDist2, avDist3 = computeObjectiveDistance(C_df_original[range(topK)], C_df[range(topK)])
avDist1, avDist2 = computeObjectiveDistance(C_df_original[range(topK)], C_df[range(topK)])

#precisionList.append(avPrecision)
#recallList.append(avRecall)
dist1List.append(avDist1)
dist2List.append(avDist2)
# dist3List.append(avDist3)
BERList.append(ber)
numberOfPeopleMovedList.append(numberOfPeopleMoved)


# In[11]:

# algorithm1
# move l b1 operation
mTimerStart = timer()
for ii in range(int(l)):
    
    v0_Hungary_Index = womanToBeMovedTable[ii][0]
    v1_Hungary_Index = womanToBeMovedTable[ii][1]
    
    sourceVectorIndex = int(v1_Hungary_Index.split('_')[0])
    destinationVectorIndex = int(v0_Hungary_Index.split('_')[0])
    
    sourceVector = summaryDF.loc[sourceVectorIndex].values[:topK].astype(int)
    destinationVector = summaryDF.loc[destinationVectorIndex].values[:topK].astype(int)
    
    womanSubjectToMoveDf = C_df.loc[C_df[C_df[range(topK)] == sourceVector].dropna(thresh=topK).index]
    womanSubjectToMoveIndex = womanSubjectToMoveDf[womanSubjectToMoveDf['b'] == 1].sample().index[0]
    
    C_df.loc[womanSubjectToMoveIndex,range(topK)] = destinationVector
    
    summaryDF.loc[sourceVectorIndex,'H(1,y)'] = summaryDF.loc[sourceVectorIndex,'H(1,y)'] - 1
    summaryDF.loc[sourceVectorIndex,'size'] = summaryDF.loc[sourceVectorIndex,'size'] - 1
    summaryDF.loc[sourceVectorIndex,'s(y)'] = summaryDF.loc[sourceVectorIndex,'s(y)'] + (1./b1)
    summaryDF.loc[sourceVectorIndex,'q(y)'] = summaryDF.loc[sourceVectorIndex,'q(y)'] - 1
    
    summaryDF.loc[destinationVectorIndex,'H(1,y)'] = summaryDF.loc[destinationVectorIndex,'H(1,y)'] + 1
    summaryDF.loc[destinationVectorIndex,'size'] = summaryDF.loc[destinationVectorIndex,'size'] + 1
    summaryDF.loc[destinationVectorIndex,'s(y)'] = summaryDF.loc[destinationVectorIndex,'s(y)'] - (1./b1)
    summaryDF.loc[destinationVectorIndex,'p(y)'] = summaryDF.loc[destinationVectorIndex,'p(y)'] - 1
    
    numberOfPeopleMoved = numberOfPeopleMoved + 1
    #if womanSubjectToMoveIndex in userTestRatedItems.keys():
    #    avPrecision, avRecall = calculateChangeInMetrics(sourceVector, destinationVector, womanSubjectToMoveIndex, C_df, userTestRatedItems, precisionList[-1], recallList[-1])
    #else:
    #    avPrecision, avRecall = precisionList[-1], recallList[-1]
        
    ber = BERList[-1] + (1./(2*b1))
    # avDist1, avDist2, avDist3 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1], dist3List[-1])
    avDist1, avDist2 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1])

    deltaBerList.append(1./(2*b1))
    #precisionList.append(avPrecision)
    #recallList.append(avRecall)
    dist1List.append(avDist1)
    dist2List.append(avDist2)
    # dist3List.append(avDist3)
    BERList.append(ber)
    numberOfPeopleMovedList.append(numberOfPeopleMoved)

mTimerEnd = timer()
print 'Moving woman took: ' + str(mTimerEnd - mTimerStart)


# In[12]:

summaryDF = computeSummaryDF(C_df)


# In[13]:

sigma_w, d, k1, k2, k, _2_b0d, l = compute_BER_d_p_q_k_l_beforeMovingMan(summaryDF)

d = epsilon - sigma_w
if d <= 0:
    print 'C is fair'
    C_df.to_csv(C_df_Out_Path)
    sys.exit()
    


# In[14]:

sigma_w, d, k1, k2, k, _2_b0d, l
berAfterMovingWoman = sigma_w
numberOfMansToBeMoved = l

print 'berAfterMovingWoman = ' + str(berAfterMovingWoman)
print 'BER double check = ' + str(BERList[-1])
print 'numberOfMansToBeMoved = ' + str(numberOfMansToBeMoved)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[15]:

if l > 0: 
    # compute V1 and V0
    movementPossibleY_1 = summaryDF[(summaryDF['Y_x'] == 1) & (summaryDF['q(y)'] > 0)]
    movementPossibleY_0 = summaryDF[(summaryDF['Y_x'] == 0) & (summaryDF['p(y)'] > 0)]

    V1 = []
    for i in movementPossibleY_1.index:
        q_z = int(movementPossibleY_1.loc[i, 'q(y)'])
        for q_z_index in range(q_z):
            V1.append(str(int(i))+'_'+ str(q_z_index))

    V0 = []
    for i in movementPossibleY_0.index:
        p_y = int(movementPossibleY_0.loc[i, 'p(y)'])
        for p_y_index in range(p_y):
            V0.append(str(int(i))+'_'+ str(p_y_index))

    print '|V0| = ' + str(len(V0))
    print '|V1| = ' + str(len(V1))
    
    summaryDF_ForCartesian = summaryDF[range(topK)]
    summaryDF_ForCartesian_Dict = summaryDF_ForCartesian.T.to_dict()
    for key in summaryDF_ForCartesian_Dict.iterkeys():
        summaryDF_ForCartesian_Dict[key] = set(summaryDF_ForCartesian_Dict[key].values())

    flowStart = timer()

    V_dictionary_orj_to_target = {}
    V_dictionary_target_to_orj = {}

    c=2
    for i in V0:
        V_dictionary_orj_to_target[i] = c
        V_dictionary_target_to_orj[c] = i
        c = c +1

    for i in V1:
        V_dictionary_orj_to_target[i] = c
        V_dictionary_target_to_orj[c] = i
        c = c +1

    from ortools.graph import pywrapgraph

    v_s = 0
    v_t = 1

    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []

    supplies = [l,-l]

    for i_v0 in V0:
        start_nodes.append(v_s)
        end_nodes.append(V_dictionary_orj_to_target[i_v0])
        capacities.append(1)
        unit_costs.append(0)

    for i_v1 in V1:
        start_nodes.append(V_dictionary_orj_to_target[i_v1])
        end_nodes.append(v_t)
        capacities.append(1)
        unit_costs.append(0)
    
    for i_v0 in V0:
        y0i = i_v0.split('_')[0]
        for i_v1 in V1:
            y1i = i_v1.split('_')[0]

            start_nodes.append(V_dictionary_orj_to_target[i_v0])
            end_nodes.append(V_dictionary_orj_to_target[i_v1])
            capacities.append(1)
            cost = len(summaryDF_ForCartesian_Dict[int(y0i)] - summaryDF_ForCartesian_Dict[int(y1i)])*2
            unit_costs.append(cost)
    
    for i in range(len(V0+V1)):
        supplies.append(0)

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],capacities[i], unit_costs[i])

    # Add node supplies.
    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    resultList = []
    # Find the minimum cost flow between node 0 and node 4.
    manToBeMovedTable = []
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        min_cost_flow.Solve()
        print('Minimum cost:', min_cost_flow.OptimalCost())
        for i in range(min_cost_flow.NumArcs()):
            if (min_cost_flow.Flow(i) != 0) and (min_cost_flow.Tail(i) != 0) and (min_cost_flow.Head(i) != 1):
                manToBeMovedTable.append([V_dictionary_target_to_orj[min_cost_flow.Tail(i)],V_dictionary_target_to_orj[min_cost_flow.Head(i)]])
    else:
        print('There was an issue with the min cost flow input.')

    flowEnds = timer()

    print "Min cost flow algorithm calculation time (seconds): " + str(flowEnds - flowStart)


# In[ ]:




# In[ ]:




# In[ ]:

# algorithm1
# move l b1 operation
movingClockStart = timer()
for ii in range(numberOfMansToBeMoved):
    
    v0_Hungary_Index = manToBeMovedTable[ii][0]
    v1_Hungary_Index = manToBeMovedTable[ii][1]
    
    sourceVectorIndex = int(v0_Hungary_Index.split('_')[0])
    destinationVectorIndex = int(v1_Hungary_Index.split('_')[0])
    
    sourceVector = summaryDF.loc[sourceVectorIndex].values[:topK].astype(int)
    destinationVector = summaryDF.loc[destinationVectorIndex].values[:topK].astype(int)
    
    manSubjectToMoveDf = C_df.loc[C_df[C_df[range(topK)] == sourceVector].dropna(thresh=topK).index]
    manSubjectToMoveIndex = manSubjectToMoveDf[manSubjectToMoveDf['b'] == 0].sample().index[0]
    
    C_df.loc[manSubjectToMoveIndex,range(topK)] = destinationVector
    
    summaryDF.loc[sourceVectorIndex,'H(0,y)'] = summaryDF.loc[sourceVectorIndex,'H(0,y)'] - 1
    summaryDF.loc[sourceVectorIndex,'size'] = summaryDF.loc[sourceVectorIndex,'size'] - 1
    summaryDF.loc[sourceVectorIndex,'s(y)'] = summaryDF.loc[sourceVectorIndex,'s(y)'] + (1./b0)
    summaryDF.loc[sourceVectorIndex,'p(y)'] = summaryDF.loc[sourceVectorIndex,'p(y)'] - 1
    
    summaryDF.loc[destinationVectorIndex,'H(0,y)'] = summaryDF.loc[destinationVectorIndex,'H(0,y)'] + 1
    summaryDF.loc[destinationVectorIndex,'size'] = summaryDF.loc[destinationVectorIndex,'size'] + 1
    summaryDF.loc[destinationVectorIndex,'s(y)'] = summaryDF.loc[destinationVectorIndex,'s(y)'] - (1./b0)
    summaryDF.loc[destinationVectorIndex,'q(y)'] = summaryDF.loc[destinationVectorIndex,'q(y)'] - 1
    
    numberOfPeopleMoved = numberOfPeopleMoved + 1
    
    #if manSubjectToMoveIndex in userTestRatedItems.keys():
    #    avPrecision, avRecall = calculateChangeInMetrics(sourceVector, destinationVector, manSubjectToMoveIndex, C_df, userTestRatedItems, precisionList[-1], recallList[-1])
    #else:
    #    avPrecision, avRecall = precisionList[-1], recallList[-1]
        
    ber = BERList[-1] + (1./(2*b0))
    # avDist1, avDist2, avDist3 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1], dist3List[-1])
    avDist1, avDist2 = calculateChangeInDistance(sourceVector, destinationVector, dist1List[-1], dist2List[-1])

    deltaBerList.append(1./(2*b0))
    #precisionList.append(avPrecision)
    #recallList.append(avRecall)
    dist1List.append(avDist1)
    dist2List.append(avDist2)
    # dist3List.append(avDist3)
    BERList.append(ber)
    numberOfPeopleMovedList.append(numberOfPeopleMoved)
    
movingClockEnd = timer()
pS = "Moving man takes that much second: " + str(movingClockEnd - movingClockStart)
print pS


# In[ ]:

sigma_wm = compute_BER(summaryDF)
finalBer = sigma_wm
print 'finalBer = ' + str(finalBer)
print 'BER double check = ' + str(BERList[-1])


# In[ ]:

if epsilon > sigma_wm:
    print 'C is still not fair'
    print sigma_wm
    
else:
    print 'C is fair'
    print sigma_wm


# In[ ]:

C_df.to_csv(C_df_Out_Path)



