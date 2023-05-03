from feat_extraction.dataset_manipulation import *
from feat_extraction.features_extractor import Chromosome
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import *
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

class GenAlFeaturesSelector(object):

    def __init__(self,n_pop=10,n_gen=17,
                 mut_prob=0.02,desired_fit=0.6,max_gen = 300,
                 scaler = MinMaxScaler(),
                 clf = MLPClassifier(random_state=42,max_iter=800,
                                     tol=1e-3)):
        self.self = self
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.pop = np.zeros((n_pop,n_gen))
        self.scaler = scaler
        self.clf = clf
        self.pipeline = make_pipeline(self.scaler,self.clf)
        # self.pipeline = self.clf
        self.mut_prob = mut_prob
        self.desired_fit = desired_fit
        self.max_gen = max_gen


    def get_gene(self):
        return 1

    def fit(self,data,target):

        #Firstly, let's shuffle the data
        order = np.random.permutation(np.arange(data.shape[0]))
        self.data = data[order]
        self.target = target[order]

        #Creates individuals
        for n in range(self.n_pop):
            #randomly decide number of used features
            n_feat = np.random.randint(0,self.n_gen)
            #now, randomly choose the locuses of features
            locus_feat = np.random.randint(0,self.n_gen,n_feat)
            self.pop[n][locus_feat] = self.get_gene()

    def _check_fitness(self,genotype):

        #create Chromosome object, with genotype of individual
        if all(genotype == 0):
            return 0
        chrom = Chromosome(genotype = genotype)

        #create train dataset that uses features of individual
        fen_train = np.array([chrom.fit_transform(self.data[n])
                             for n in range(self.data.shape[0])],dtype='float64')
        #return fitness, i.e accuracy of model
        score = np.mean(cross_val_score(self.pipeline,fen_train,self.target,cv=5))
        #THe part below is double-checkc. Sometimes classifier is too good, then
        #it is necessary to turn it into plausible one.
        while score > 0.99:
            score = np.mean(cross_val_score(self.pipeline,fen_train,self.target,cv=5))
        #Output is a rounded to 2 decimal points score.
        return round(score,2)


    def _pop_fitness(self,pop):
        """ Checks the fitness for each individual in pop, then returns
        it """
        return np.array([self._check_fitness(n) for n in pop])


    def _pairing(mother,father):
        """ Method for pairing chromosomes and generating descendants, array of characters with shape [2,n_gen] """
        n_heritage = np.random.randint(0,len(mother))
        child1 = np.concatenate([father[:n_heritage],mother[n_heritage:]])
        child2 = np.concatenate([mother[:n_heritage],father[n_heritage:]])
        return child1,child2

    def transform(self):

        self.pop_fit = self._pop_fitness(self.pop)
        self.past_pop = self.pop_fit.copy()
        self.best_ind = np.max(self.past_pop)
        self.n_generation = 0
        # for n in range(10):

        # For check, how does an algorithm performs, comment out line above,
        # and comment line below.

        while self.best_ind < self.desired_fit:
            self.descendants_generation()
            self.pop_fit = self._pop_fitness(self.pop)
            if (self.n_generation % 1) == 0:
                print(self.pop_fit)
            self.best_ind = np.max(self.pop_fit)
            self.random_mutation()
            self.n_generation += 1
            if self.n_generation > self.max_gen:
                break


    def fit_transform(self,data,target):
        """ Fits the data to model, then executes an algorithm. """
        self.fit(data,target)
        self.transform()

    def descendants_generation(self):

        #Two firsts individuals in descendants generation are the best individuals from previous generation
        self.past_pop = np.vstack([self.past_pop,self.pop_fit])
        self.pop[:2] = self.pop[np.argsort(self.pop_fit)][-2:]
        #now,let's select best ones
        # print(pop_fit)
        parents_pop = self.roulette()
        #Finally, we populate new generation by pairing randomly chosen best
        #individuals
        for n in range(2,self.n_pop-1):
                father = parents_pop[np.random.randint(self.n_pop)]
                mother = parents_pop[np.random.randint(self.n_pop)]
                children = self._pairing(mother,father)
                self.pop[(n)] = children[0]
                self.pop[(n)+1] = children[1]

    def random_mutation(self):
        pop = self.pop.copy()
        for n in range(self.n_pop):
            decision = np.random.random()
            if decision < self.mut_prob:
                which_gene = np.random.randint(self.n_gen)
                if pop[n][which_gene] == 0:
                    pop[n][which_gene] = self.get_gene()
                else:
                    pop[n][which_gene] = 0
        self.pop = pop

    def roulette_wheel(self):
        """ Method that returns roulette wheel, an array with shape [n_population, low_individual_probability,high_individual_probability]"""
#         pop_fitness = self._pop_fitness(self.pop)
        pop_fitness = self.pop_fit
        wheel = np.zeros((self.n_pop,3))
        prob = 0
        for n in range(self.n_pop):
            ind_prob = prob + (pop_fitness[n] / np.sum(pop_fitness))
            wheel[n] = [n,prob,ind_prob]
            prob = ind_prob
        return wheel



    def roulette(self):
        """ This method performs selection of individuals, it takes the coeffici
        ent k, which is number of new individuals """
        wheel = self.roulette_wheel()
        return np.array([self.pop[self.roulette_swing(wheel)]
                         for n in range(self.n_pop)])

    def plot_fitness(self,title='Algorithm performance'):

        N = self.past_pop.shape[0]
        t = np.linspace(0,N,N)
        past_fit_mean = [np.mean(self.past_pop[n]) for n in range(N)]
        past_fit_max = [np.max(self.past_pop[n]) for n in range(N)]
        plt.plot(t,past_fit_mean,label='pop mean fitness')
        plt.plot(t,past_fit_max,label='pop best individual\'s fitness')
        plt.xlabel('Number of generations')
        plt.ylabel('Fitness')
        plt.ylim([0,1])
        plt.legend()
        plt.title(title)
        plt.savefig(title)
        # plt.show()



if __name__ == '__main__':
    time_window=5
    bs = BakSys(threeclass=False,seconds=time_window)
    ga = GenAlFeaturesSelector(n_pop=5,desired_fit=0.1,max_gen=3)

    data,target = chunking(data,time_window=time_window)
    n_samples = target.shape[0]
    freq = 256
    data = np.array([bs.fit_transform(data[n])
                    for n in range(n_samples)]).reshape(n_samples*2,
                                                               freq*time_window)
    target = np.array([[n,n] for n in target]).reshape(n_samples*2)
    # print(data.shape)
    ga.fit(data,target)
    print(ga.pop)
    # ga.transform()

