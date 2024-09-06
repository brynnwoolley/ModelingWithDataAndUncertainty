import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse


class NMFRecommender:

    def __init__(self,random_state=15,rank=2,maxiter=200,tol=1e-3):
        """
        Save the parameter values as attributes.
        """
        # initialize
        self.random_state = random_state
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol

    def _initialize_matrices(self, m, n):
        """
        Initialize the W and H matrices.
        
        Parameters:
            m (int): the number of rows
            n (int): the number of columns
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        # set the random seed 
        np.random.seed(self.random_state)

        # initialize W and H using np.random.random
        W = np.random.random((m, self.rank))
        H = np.random.random((self.rank, n))

        return W, H

    def _compute_loss(self, V, W, H):
        """
        Compute the loss of the algorithm according to the 
        Frobenius norm.
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        """
        # compute frobenius norm of V - WH
        return np.linalg.norm(V - (W@H), ord='fro')

    def _update_matrices(self, V, W, H):
        """
        The multiplicative update step to update W and H
        Return the new W and H (in that order).
        
        Parameters:
            V ((m,n) array): the array to decompose
            W ((m,k) array)
            H ((k,n) array)
        Returns:
            New W ((m,k) array)
            New H ((k,n) array)
        """
        # update W and H
        H_new = H*((W.T@V) / (W.T@(W@H)))
        W_new = W*((V@H_new.T) / (W@(H_new@H_new.T)))

        return W_new, H_new

    def fit(self, V):
        """
        Fits W and H weight matrices according to the multiplicative 
        update algorithm. Save W and H as attributes and return them.
        
        Parameters:
            V ((m,n) array): the array to decompose
        Returns:
            W ((m,k) array)
            H ((k,n) array)
        """
        # initialize W and H
        W, H = self._initialize_matrices(V.shape[0], V.shape[1])
        i = 0

        while True:
            # update W and H
            W, H = self._update_matrices(V, W, H)

            # calculate loss
            loss = self._compute_loss(V, W, H)

            # check for convergence
            if loss < self.tol or i > self.maxiter:
                break
            i += 1
        # save & store W and H as attributes
        self.W = W
        self.H = H

        return W, H

    def reconstruct(self):
        """
        Reconstruct and return the decomposed V matrix for comparison against 
        the original V matrix. Use the W and H saved as attrubutes.
        
        Returns:
            V ((m,n) array): the reconstruced version of the original data
        """
        return self.W @ self.H

def prob4(rank=2):
    """
    Run NMF recommender on the grocery store example.
    
    Returns:
        W ((m,k) array)
        H ((k,n) array)
        The number of people with higher component 2 than component 1 scores
    """
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])
                  
    # construct NMFRecommender & fit W and H
    nmf = NMFRecommender(rank=rank)
    W, H = nmf.fit(V)

    # calculate # of people w/ higher component 2 than component 1 scores
    num_higher = np.sum(H[1] > H[0])

    return W, H, num_higher.astype(float)


def prob5(filename='artist_user.csv'):
    """
    Read in the file `artist_user.csv` as a Pandas dataframe. Find the optimal
    value to use as the rank as described in the lab pdf. Return the rank and the reconstructed matrix V.
    
    Returns:
        rank (int): the optimal rank
        V ((m,n) array): the reconstructed version of the data
    """
    # load data & calculate benchmark value
    df = pd.read_csv(filename, index_col=0)
    orig_V = df.values
    benchmark = np.linalg.norm(df.values, ord='fro')*0.0001

    # iterate & find the optimal rank
    for rank in range(10, 15):
        model = NMF(n_components=rank, init='random', random_state=0)
        W = model.fit_transform(orig_V)
        H = model.components_

        V = W @ H

        # calculate RMSE
        loss = np.sqrt(mse(orig_V, V))

        # check for convergence
        if loss < benchmark:
            return rank, V


def discover_weekly(userid, V):
    """
    Create the recommended weekly 30 list for a given user.
    
    Parameters:
        userid (int): which user to do the process for
        V ((m,n) array): the reconstructed array
        
    Returns:
        recom (list): a list of strings that contains the names of the recommended artists
    """
    recs = {}
    # get users row from V & artists names from the dataframe
    user_row = V[userid - 2]
    artists = pd.read_csv('artists.csv', index_col=0).values
    artists_users = pd.read_csv('artist_user.csv', index_col=0)

    # get the # of times each artist was listened to by the user
    listened = artists_users.loc[userid].values

    for i, times_listened in enumerate(listened):
        if times_listened == 0:
            # add the artist name and the score to the dictionary
            recs[artists[i][0]] = user_row[i]

    # sort the dictionary recs by the values & return the top 30
    top_30 = sorted(recs, key=recs.get, reverse=True)[:30]

    # print the artist names
    print(top_30)
    
    