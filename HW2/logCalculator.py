# Chaehyeon Kim (cxk445); for CSDS 340 Homework Assignment 2
# For solving number 6

import math

def entropy(numList, portion):
    totalEntropy = 0
    for num in numList:
        logVal = num * math.log2(num)
        totalEntropy = totalEntropy + logVal
    return portion * (-1 * totalEntropy)
        

if __name__ == "__main__":
    percentages = [1/3, 2/3]
    portion = 3/10
    term = entropy(percentages, portion)
    # print(term)

    # totalList = [[[5/10, 3/10, 1/10, 1/10], 1], 
    #              [[1/3, 2/3], 3/10], 
    #              [[2/4, 1/4, 1/4], 4/10], 
    #              [[2/3, 1/3], 3/10]]
    totalList = [[[1/2, 1/2], 1],
                 [[1/2, 1/2], 2/4],
                 [[1/2, 1/2], 2/4]]
    totalVal = 0
    for i in range(len(totalList)):
        percentages = totalList[i][0]
        portion = totalList[i][1]
        term = entropy(percentages, portion)
        if (i==0):
            totalVal = term
        else:
            totalVal = totalVal - term
    print(totalVal)
