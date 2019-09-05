import numpy as np
from kmeans import KMeans

class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def get_det_inv(self):
        # cal det(var) and inv(var)
        weit_collect = []
        varinv_collect = []
        D = self.variances.shape[1]
        for i in range(self.n_cluster):
            var = self.variances[i]
            pai = self.pi_k[i]
            rank = np.linalg.matrix_rank(var)
            while rank < D:
                var += (10 ** -3) * np.identity(D)
                rank = np.linalg.matrix_rank(var)
            temp_weit = pai / pow((((2 * np.pi)**D) * np.linalg.det(var)), 0.5)
            weit_collect.append(temp_weit)
            varinv = np.linalg.inv(var)
            varinv_collect.append(varinv)
        return weit_collect, varinv_collect

    def density(self,x, weit, varinv, k_mean):
        temp = - 0.5 * np.linalg.multi_dot([x - k_mean, varinv, (x - k_mean).T])
        re = weit * np.exp(temp)
        return re



    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')

            #run kmeans
            temp = KMeans.fit(self, x)
            self.means = temp[0]
            k_label = temp[1]

            #cal variance matrix
            idx = (k_label == 0)
            k_x = x[(k_label == 0)]
            k_n = len(idx)
            k_var = np.dot(k_x.T, k_x) / k_n
            k_var = np.reshape(k_var, (1, D, D))
            k_pik = [k_n]

            for i in range(1, self.n_cluster):
                idx = (k_label == i)
                k_x = x[idx]
                k_n = len(idx)
                temp_k_var = np.dot(k_x.T, k_x) / k_n
                temp_k_var = np.reshape(temp_k_var, (1, D, D))
                k_var = np.concatenate((k_var, temp_k_var), axis = 0)
                k_pik.append(k_n)
            self.variances = k_var
            self.pi_k = np.array(k_pik) / N
            # print(self.variances.shape,'kmeans')


            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')
            self.means = np.random.rand(self.n_cluster, D)
            self.variances = np.tile(np.identity(D), (self.n_cluster,1,1))
            # print(self.variances.shape)
            self.pi_k = np.full((self.n_cluster,), 1 / self.n_cluster)

            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - Use EM to learn the means, variances, and pi_k and assign them to self
        # - Update until convergence or until you have made self.max_iter updates.
        # - Return the number of E/M-Steps executed (Int) 
        # Hint: Try to separate E & M step for clarity
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement fit function (filename: gmm.py)')

        #cal log-likelihood


        # l = 0
        # for i in range(x.shape[0]):
        #     temp_l = 0
        #     for j in range(self.n_cluster):
        #         temp_re = self.density(x[i],self.pi_k[j],self.variances[j],self.means[j])
        #         temp_l += temp_re
        #     l += np.log(temp_l)
        # print(x.shape,'xshape')
        l = self.compute_log_likelihood(x)
        count = 0
        weit_collect, varinv_collect = self.get_det_inv()
        while count != self.max_iter:
            #E step
            gama = np.zeros((N, self.n_cluster))
            for i in range(N):
                for j in range(self.n_cluster):
                    gama[i,j] = self.density(x[i], weit_collect[j], varinv_collect[j], self.means[j])

            gama_sum = np.expand_dims(np.sum(gama, axis = -1), axis = -1)
            gama = gama / gama_sum


            #M step:
            N_k = np.expand_dims(np.sum(gama, axis = 0), axis = -1)
            #1. update means
            self.means = np.dot(gama.T, x) / N_k
            x_0 = (x - self.means[0])

            var = np.dot((np.expand_dims(gama[:,0], axis = -1)*x_0).T, x_0) / N_k[0]
            var = var.reshape(1,D,D)
            for i in range(1, self.n_cluster):
                temp_x_0 = (x - self.means[i])

                temp_var = (np.dot((np.expand_dims(gama[:,i], axis = -1) * temp_x_0).T, temp_x_0) / N_k[i]).reshape(1,D,D)
                var = np.concatenate((var, temp_var), axis = 0)
            self.variances = var
            self.pi_k = N_k / N

            #cal new l
            # new_l = 0
            # for i in range(x.shape[0]):
            #     temp_l = 0
            #     for j in range(self.n_cluster):
            #         temp_re = self.density(x[i], self.pi_k[j], self.variances[j], self.means[j])
            #         temp_l += temp_re
            #     new_l += np.log(temp_l)
            new_l = self.compute_log_likelihood(x)

            if abs(l - new_l) < self.e:
                return count

            l = new_l
            # print(count)
            count+=1



        return count

        # DONOT MODIFY CODE BELOW THIS LINE
    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        # print(x.shape)
        assert len(x.shape) == 2, 'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k
            # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')

        weit_collect, varinv_collect = self.get_det_inv()

        new_l = 0
        for i in range(x.shape[0]):
            temp_l = 0
            for j in range(self.n_cluster):
                temp_re = self.density(x[i], weit_collect[j], varinv_collect[j], self.means[j])
                temp_l += temp_re
            new_l += np.log(temp_l)

        # DONOT MODIFY CODE BELOW THIS LINE
        self.pi_k = self.pi_k.reshape(self.n_cluster, )
        # print(new_l,'new_l')
        return float(new_l)
		
    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        D = self.variances.shape[1]
        #sample k to each N
        N_k = np.random.choice(self.n_cluster, (N,1), p = self.pi_k)

        #sample Guassian Distribution
        means = self.means
        variances = self.variances
        def g_d(k):
            k = k[0]
            temp_means = means[k]
            # print(temp_means.shape, 'mean.shape')
            temp_var = variances[k]
            # print(temp_var.shape,'var_shape')
            temp_re = np.random.multivariate_normal(temp_means, temp_var)
            return temp_re

        samples = np.apply_along_axis(g_d, -1, N_k)
        samples = samples.reshape(N,D)



        # raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples        



    class Gaussian_pdf():
        def __init__(self,mean,variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            # # cal det(var) and inv(var)
            # weit_collect = []
            # varinv_collect = []
            # D = x.shape[1]
            # for i in range(self.n_cluster):
            #     var = self.variances[i]
            #     pai = self.pi_k[i]
            #     rank = np.linalg.matrix_rank(var)
            #     while rank < D:
            #         var += (10 ** -3) * np.identity(D)
            #         rank = np.linalg.matrix_rank(var)
            #     temp_weit = pai / pow(((2 * np.pi) * np.linalg.det(var)), 0.5)
            #     weit_collect.append(temp_weit)
            #     varinv = np.linalg.inv(var)
            #     varinv_collect.append(varinv)
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self,x):
            '''
                Input: 
                    x: a 1 X D numpy array representing a sample
                Output: 
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint: 
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''
            #TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            raise Exception('Impliment Guassian_pdf getLikelihood')
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
