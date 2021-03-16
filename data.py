import pandas as pd
import numpy as np

from sklearn import model_selection


class Data:
    def __init__(self, path=None):
        if path is not None:
            self.file = pd.read_csv(path)
        else:
            self.file = pd.DataFrame()
        self.labels = []
        self.save_features = pd.DataFrame()

    def preprocess(self, knn_flag=False, rocchio_flag=False):
        """
        Prepares the dataframe according to the given instruction (different classifiers), prepares the labels, and the
        relevant features.
        :param knn_flag: Determines whether to prepare the data for knn classifier
        :param rocchio_flag: Determines whether to prepare the data for rocchio classifier
        """
        # *******************************************************************
        # Dropping irrelevant data
        self.file = self.file.drop(columns=['content_rating', 'movie_imdb_link', 'plot_keywords'])
        self.file = self.file.replace('', np.nan).dropna()
        self.file = self.file.drop_duplicates('movie_title')
        self.file = self.file.drop(columns='movie_title')
        # ********************************************
        categories_features = ['actor_1_name', 'actor_2_name', 'actor_3_name',
                               'director_name', 'genres', 'country', 'color', 'language']
        features_to_drop2 = []
        if knn_flag:
            features_to_drop2 = ['color', 'language']

        if rocchio_flag:
            features_to_drop2 = ['color', 'language']

        self.file = self.file.drop(columns=features_to_drop2)
        temp_list = [x for x in categories_features if x not in features_to_drop2]
        categories_features = temp_list
        # *****************************************
        for name in categories_features:
            df1 = self.file[name].str.get_dummies()
            self.file = pd.concat([self.file, df1], axis=1)
        self.file = self.file.drop(columns=categories_features)
        self.file = self.file.groupby(self.file.columns, axis=1).sum()

        # Binary classification of the scores - labels
        self.file.loc[self.file['imdb_score'] < 7, 'imdb_score'] = 0
        self.file.loc[self.file['imdb_score'] >= 7, 'imdb_score'] = 1

        # Extracting the labels from the dataset
        self.labels = self.file['imdb_score']
        self.file = self.file.drop(columns='imdb_score')

        numerical_features = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                              'aspect_ratio', 'budget', 'cast_total_facebook_likes', 'director_facebook_likes',
                              'duration', 'facenumber_in_poster', 'gross', 'movie_facebook_likes',
                              'num_critic_for_reviews', 'num_user_for_reviews', 'num_voted_users', 'title_year']

        # *********************************************************************
        features_to_drop = []
        if knn_flag:
            features_to_drop = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                                'director_facebook_likes', 'cast_total_facebook_likes', 'movie_facebook_likes',
                                'facenumber_in_poster', 'duration', 'aspect_ratio']
        if rocchio_flag:
            features_to_drop = ['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes',
                                'director_facebook_likes', 'cast_total_facebook_likes', 'movie_facebook_likes',
                                'facenumber_in_poster', 'aspect_ratio']

        self.file = self.file.drop(columns=features_to_drop)
        temp_list = [x for x in numerical_features if x not in features_to_drop]
        numerical_features = temp_list
        # **********************************************************************

        # Normalize the values
        for word in numerical_features:
            my_std = self.file[word].std()
            my_mean = self.file[word].mean()
            self.file[word] = (self.file[word] - my_mean) / my_std

    def split_to_k_folds(self):
        """
        Splits the data to the desired amount of folds.
        :return: The indexes of the separated data.
        """
        kf = model_selection.KFold(n_splits=5, shuffle=False, random_state=None)
        return kf.split(self.file, self.labels)

