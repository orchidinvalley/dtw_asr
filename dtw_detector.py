#! /usr/bin/env python

import numpy as np

class DtwDetector(object):


    def dtw_count(self, t, r, d=lambda x, y: np.sum(np.square(abs(x-y))),nww=10000):
        t, r = np.array(t), np.array(r)
        m, n = len(t), len(r)
        cost = np.ones((m, n))


        cost[0, 0]=d(t[0], r[0])
        for i in range(1, m):
            cost[i, 0] = cost[i-1, 0]+d(t[i], r[0])

        for j in range(1, n):
            cost[0, j] = cost[0, j-1]+d(t[0], r[j])

        for i in range(1,m):
            for j in range(1,n):
                choices = cost[i-1,j-1],cost[i,j-1],cost[i-1,j]
                cost[i,j]=min(choices)+d(t[i],r[j])


        return int(cost[-1,-1])




