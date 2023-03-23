import numpy as np
def calculate_prior(df, Y):
            classes = sorted(list(df[Y].unique()))
            prior = []
            for i in classes:
                prior.append(len(df[df[Y]==i])/len(df))
            return prior

def calculate_likelihood_gaussian(df, feat_name, feat_val, Y, label):
        feat = list(df.columns)
        df = df[df[Y]==label]
        mean, std = df[feat_name].mean(), df[feat_name].std()
        p_x_given_y = (1 / (np.sqrt(2 * np.pi) * std)) *  np.exp(-((feat_val-mean)**2 / (2 * std**2 )))
        return p_x_given_y
class My_Naive_Bayes:
    # def __init__(self) -> None:
    #     self.df = None

    def naive_bayes_gaussian(df, X, Y):
        features = list(df.columns)[:-1]
        prior = calculate_prior(df, Y)

        Y_pred = []
        for x in X:
            labels = sorted(list(df[Y].unique()))
            likelihood = [1]*len(labels)
            for j in range(len(labels)):
                for i in range(len(features)):
                    likelihood[j] *= calculate_likelihood_gaussian(df, features[i], x[i], Y, labels[j])

            post_prob = [1]*len(labels)
            for j in range(len(labels)):
                post_prob[j] = likelihood[j] * prior[j]
            #P(Y|X)
            Y_pred.append(np.argmax(post_prob))
        return Y_pred

