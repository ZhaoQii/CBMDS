# Original Paper:
# DeSarbo, Wayne S., Rajdeep Grewal, and Crystal J. Scott. "A clusterwise bilinear multidimensional scaling methodology for simultaneous segmentation and positioning analyses." Journal of Marketing Research 45, no. 3 (2008): 280-292.

# Main function to run the proposed method for simultaneous customer segmentation and product positioning analysis
# Notes: does not support reparameterization module, which means it requires S >= T as noted in the original paper

def CBMDS_JMR2008(Delta, S, T):
    # N: the number of all respondents
    # J: the number of all competing alternatives (e.g., products or brands)
    # S: the number of expected customer segments
    # T: the dimensionality of the derived joint space
    # Delta is the pre-processed preference/intention/attitude rating data. Dimension: N-by-J
    
    N, J = Delta.shape
    
    # initialization
    b = 0
    X = np.random.normal(size = [J, T])
    Y = np.random.normal(size = [S, T])
    P = np.random.choice(a = [0, 1], size = [N, S])
    P[np.where(np.sum(P, 1) == 0)[0], 0] = 1
    
    # estimates of ratings
    Delta_hat = np.matmul(np.matmul(P, Y), X.transpose())
    
    # fitting performance
    VAF_last = 1 -  np.sum((Delta - Delta_hat) ** 2) / np.sum((Delta - Delta.mean()) ** 2)
    VAF = VAF_last + 1

    # the estimation iteration 
    P_all_can = np.array(list(itertools.product([0, 1], repeat = S)))[1:]
    iter = 0
    while VAF - VAF_last > 0.0001:
        iter += 1
        # print convergence cretirion
        print('Change of VAF at iter {}:'.format(iter), VAF - VAF_last)
        print('VAF at iter {}:'.format(iter), VAF)

        # update Delta_star
        Delta_star = Delta - b

        # 1 update X
        temp1 = np.matmul(P, Y)
        temp2 = np.linalg.inv(np.matmul(temp1.transpose(), temp1))
        X = np.matmul(np.matmul(Delta_star.transpose(), temp1), temp2)

        # 2 update Y
        temp1 = np.linalg.inv(np.matmul(X.transpose(), X))
        temp2 = np.linalg.inv(np.matmul(P.transpose(), P))
        Y = np.matmul(temp1, np.matmul(np.matmul(np.matmul(X.transpose(), Delta_star.transpose()), P), temp2)).transpose()

        # 3 update P
        min_temps = np.zeros(2 ** S - 1)
        for i in range(N):
            min_temps = np.asarray([np.matmul((Delta_star[i] - np.matmul(P_all_can[s], np.matmul(Y, X.transpose()))).transpose(), (Delta_star[i] - np.matmul(P_all_can[s], np.matmul(Y, X.transpose())))) for s in range((2 ** S - 1))])
            P[i] = P_all_can[min_temps.argmin()]

        # 4 update b
        Delta_hat = np.matmul(np.matmul(P, Y), X.transpose())
        K = np.vstack((np.ones(N * J), Delta_hat.flatten('F'))).transpose()
        L = Delta.flatten('F')
        b = np.matmul(np.matmul(np.linalg.inv(np.matmul(K.transpose(), K)), K.transpose()), L)[0]

        VAF_last = VAF.copy()
        VAF = 1 - np.sum((Delta - Delta_hat) ** 2) / np.sum((Delta - Delta.mean()) ** 2)
    return(P, X, Y, b, Delta_hat)
