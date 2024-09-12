import os
import pickle
import numpy as np
from scipy.linalg import LinAlgError
import cv2
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io
from skimage.segmentation import quickshift, mark_boundaries
from skimage.measure import regionprops
import copy
import random
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from skimage import filters
import pandas as pd
import warnings
import tensorflow as tf
import pickle
import sys
from scipy.stats import kendalltau

from matplotlib import pyplot as plt
import time
from sklearn.utils import resample
from scipy.stats import norm, gaussian_kde
from functools import partial
from sklearn.neighbors import KernelDensity
import csv

class SliceExplainer:

    def __init__(self, image_path, segments, model,
                 target_img_size, preprocess):
        """
        Initialize the SliceExplainer instance.

        Parameters:
        - image_path (str): Path to the image to be explained.
        - segments (array): Segments/superpixels of the image.
        - model (tf.keras.Model): Pre-trained model to generate predictions.
        - target_img_size (tuple): Target size for the image.
        - preprocess (function, optional): Preprocessing function for the image.

        Attributes:
        - img (array): Processed image.
        - model (tf.keras.Model): Model to generate predictions.
        - superpixels (array): Segments/superpixels of the image.
        - top_pred_class (int): Top predicted class index for the original image.
        - prob0 (float): Probability of the top predicted class for the original image.
        - train_mat (array): Training matrix for explanation.
        - train_mat_sel_idx (array): Index selector for the training matrix.
        - sigma (float or None): Sigma value for Gaussian blur.
        """

        img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)

        if model.name != "vitp16":
            img = preprocess(img)

        #img = preprocess(img)
        self.img = img
        self.model = model
        self.superpixels = segments
        self.top_pred_class, self.prob0 = self._get_pred0()
        self.train_mat = np.append(np.ones(np.unique(self.superpixels).shape[0]), self.prob0)
        self.train_mat = self.train_mat.reshape((1, len(self.train_mat)))
        self.train_mat_sel_idx = np.zeros(np.unique(self.superpixels).shape[0])
        self.sigma = None

    @classmethod
    def from_arrays(cls, img_array, segments_array, model):
        """
        Alternative constructor to create a SliceExplainer instance from image and segments arrays.

        Parameters:
        - img_array (array): The image array.
        - segments_array (array): Segments/superpixels of the image.
        - model (tf.keras.Model): Pre-trained model to generate predictions.
        - target_img_size (tuple): Target size for the image.
        - preprocess (function, optional): Preprocessing function for the image.

        Returns:
        - SliceExplainer instance.
        """
        instance = cls.__new__(cls)
        instance.img = img_array
        instance.model = model
        instance.superpixels = segments_array
        instance.top_pred_class, instance.prob0 = instance._get_pred0()
        instance.train_mat = np.append(np.ones(np.unique(instance.superpixels).shape[0]), instance.prob0)
        instance.train_mat = instance.train_mat.reshape((1, len(instance.train_mat)))
        instance.train_mat_sel_idx = np.zeros(np.unique(instance.superpixels).shape[0])
        instance.sigma = None

        return instance

    def _get_pred0(self):
        if self.model.name == "vitp16":
            pred0 = self.model.predict(self.img, verbose=0)
        else:
            pred0 = self.model.predict(np.array([self.img]), verbose=0)

        top_pred_class = pred0[0].argsort()[-1:][::-1]
        prob0 = pred0[0][top_pred_class]
        return top_pred_class, prob0

    def _perturb_image(self, img, perturbation, segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image * mask[:, :, np.newaxis]
        return perturbed_image

    def _perturb_image_blur(self, img, perturbation, segments, sigma=1):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1
        mask3d = cv2.merge((mask, mask, mask))
        perturbed_image = copy.deepcopy(img)
        blur_img = filters.gaussian(img, sigma)
        perturbed_image = np.where(mask3d == np.array([0.0, 0.0, 0.0]), blur_img, img)
        return perturbed_image

    def _generate_perturbations(self, num_perturb):
        num_perturb = num_perturb - self.train_mat.shape[0]

        if num_perturb > 0:
            num_superpixels = np.unique(self.superpixels).shape[0]
            perturbations = np.random.randint(0, 2, num_perturb * num_superpixels).reshape(
                (num_perturb, num_superpixels))
        else:
            perturbations = None
        return perturbations

    def _generate_pert_images(self, perturbations):
        perturbed_imgs = []
        for pert in perturbations:
            if self.sigma == 0:
                perturbed_img = self._perturb_image(self.img, pert, self.superpixels)
            else:
                perturbed_img = self._perturb_image_blur(self.img, pert, self.superpixels, sigma=self.sigma)
            perturbed_imgs.append(perturbed_img)

        return perturbed_imgs

    def _populate_train_matrix(self, perturbations, refresh, batch_size=64):
        num_images = perturbations.shape[0]
        predictions = []
        for i in range(0, num_images, batch_size):
            try:
                batch = self._generate_pert_images(perturbations[i:i + batch_size])
            except Exception as e:
                print(f"Error in predicting batch: {e}")
                continue
            batch_predictions = self.model.predict(np.array(batch), verbose=False)
            predictions.extend(batch_predictions)

        predictions = np.array(predictions)
        train_matrix = np.hstack((perturbations, predictions[:, self.top_pred_class]))
        n_cols = perturbations.shape[1]
        if not np.all(self.train_mat[:, :n_cols] == perturbations):
            self.train_mat = np.vstack((self.train_mat, train_matrix))
        else:
            self.train_mat[:, -1] = predictions[:, self.top_pred_class].reshape(-1)

        return None

    def _get_features_stats(self, sample_size_search=False, n_iterations=10000, num_folds=5):
        simpler_model = Ridge(alpha=1, fit_intercept=True)
        train_matrix = self.train_mat
        simpler_model.fit(X=train_matrix[:, :-1], y=train_matrix[:, -1])
        coeffs = simpler_model.coef_

        # p_vals = stats.coef_pval(simpler_model, train_matrix[:,:-1], train_matrix[:,-1])

        # print(stats.coef_pval(simpler_model, train_matrix[:,:-1], train_matrix[:,-1]))

        y_pred = simpler_model.predict(train_matrix[:, :-1])
        y_mean = np.mean(train_matrix[:, -1])
        SS_res = np.sum((train_matrix[:, -1] - y_pred) ** 2)
        SS_tot = np.sum((train_matrix[:, -1] - y_mean) ** 2)
        R_squared = 1 - (SS_res / SS_tot)
        N = train_matrix.shape[0]
        p = train_matrix.shape[1] - 1
        adj_R_squared = 1 - ((1 - R_squared) * (N - 1) / (N - p - 1))

        if sample_size_search:
            X = train_matrix[:, :-1]
            y = train_matrix[:, -1]
            coeffs_bs = []
            for i in range(n_iterations):
                X_sample, y_sample = resample(X, y)
                ridge_model = Ridge(alpha=1)
                # Fit the Ridge Regression model on the bootstrap sample
                ridge_model.fit(X_sample, y_sample)

                # Store the coefficients
                coeffs_bs.append(ridge_model.coef_)

            coeffs_bs = np.array(coeffs_bs)

            sign_entropies = []
            # Loop through the columns of the coeffs DataFrame
            for column in range(coeffs_bs.shape[1]):
                # Get the data for the current column
                data = coeffs_bs[:, column]
                # Calculate the sign entropy for the current column
                sign_entropy = self._calculate_entropy(data)
                # Append the sign entropy to the list
                sign_entropies.append(sign_entropy)
            num_predictors = len(sign_entropies)
            sign_entropies = np.array(sign_entropies)
            av_sign_entropy = np.mean(sign_entropies)
            ratio_zero_entropy = np.count_nonzero(sign_entropies == 0) / (sign_entropies.shape[0])
            zero_indices = np.where(sign_entropies == 0)[0]
            non_zero_indices = str(np.where(sign_entropies != 0)[0])

            # print("non_zero_indices=", str(np.where(sign_entropies != 0)[0]))
            # print("non_zero_entropies=", str(sign_entropies[np.where(sign_entropies != 0)[0]]))

            # Calculate pairwise Kendall's Tau
            coeffs_bs_0ent = coeffs_bs[:, zero_indices]
            n_features = coeffs_bs_0ent.shape[1]
            kendalls = []
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    tau, _ = kendalltau(coeffs_bs_0ent[i, :], coeffs_bs_0ent[j, :])
                    kendalls.append(tau)
            # Calculate the mean Kendall's Tau
            kendalls = [tau if tau >= 0 else 0 for tau in kendalls]
            mean_kendall = np.mean(kendalls)

        else:
            av_sign_entropy = None
            ratio_zero_entropy = 0.0
            num_predictors = len(self.train_mat_sel_idx)
            mean_kendall = 1
            non_zero_indices = ""

        return coeffs, adj_R_squared, av_sign_entropy, ratio_zero_entropy, non_zero_indices, mean_kendall, num_predictors

    def _explain(self, sigma, num_samples, sample_size_search=False, refresh=False):
        self.sigma = sigma

        if refresh:
            self.train_mat = self.train_mat[0, :]
            self.train_mat = self.train_mat.reshape((1, len(self.train_mat)))

        perturbations = self._generate_perturbations(num_samples)
        if perturbations is None:
            perturbations = self.train_mat[:, :-1]

        # perturbed_images = self.generate_pert_images(perturbations)  # uses the sigma from self.sigma
        self._populate_train_matrix(perturbations, refresh)

        coeffs, adj_r2, av_sign_ent, ratio_zero_entropy, non_zero_indices, mean_kendall, num_predictors \
            = self._get_features_stats(sample_size_search)

        return coeffs, adj_r2, av_sign_ent, ratio_zero_entropy, non_zero_indices, mean_kendall, num_predictors

    def select_sigma(self, sigma_values=[0,0.1,0.2,0.3,0.4,0.5], test_sample_size=500, refresh=False):
        scores = []
        for sigma in sigma_values:
            _, adj_r2, _, _, _, _, _ = self._explain(sigma, test_sample_size, False, refresh)
            # coeffs, adj_r2, av_sign_ent, num_0_ent
            # score = (np.sum(p_values < 0.05)/len(p_values)) + adj_r2
            score = adj_r2
            scores.append(score)

        arr = np.array(scores)
        non_nan_indices = np.where(~np.isnan(arr))[0]
        # Sort the non-NaN values and get their indices
        sigma_indx = non_nan_indices[np.argsort(np.array(arr)[non_nan_indices])][-1]
        print(scores, "non nan selected=", sigma_values[sigma_indx])
        return sigma_values[sigma_indx]

    def _calculate_entropy(self, data):
        try:
            scipy_kernel = gaussian_kde(data)

            #  We calculate the bandwidth for later use
            optimal_bandwidth = scipy_kernel.factor * np.std(data)

            # Calculate KDE for the entire dataset
            kde = gaussian_kde(data, bw_method=optimal_bandwidth)

            # Create a range of values to represent the KDE
            x = np.linspace(np.min(data), np.max(data), 1000)

            # Evaluate the density at each point in the range
            density = kde(x)

            # Normalize the density function
            normalized_density = density / np.sum(density * (x[1] - x[0]))

            # Calculate the probabilities of positive and negative values
            positive_probability = np.sum(normalized_density[x >= 0] * (x[1] - x[0]))
            negative_probability = np.sum(normalized_density[x < 0] * (x[1] - x[0]))

            if positive_probability == 0 or negative_probability == 0:
                sign_entropy = 0
            else:
                sign_entropy = -positive_probability * np.log2(positive_probability) \
                               - negative_probability * np.log2(negative_probability)

        except LinAlgError as e:
            print(f"Warning: {e}. Returning 0 entropy.")
            sign_entropy = 0

        return sign_entropy

    def get_slice_explanations(self, sigma, n_iterations=1000, max_steps=10, num_perturb=500, tolerance_limit=3,
                               rank_stabilization=False):
        self.sigma = sigma
        tolerance_cur = 0
        final_coeffs = np.zeros((n_iterations, len(self.train_mat_sel_idx)))
        for step in range(max_steps):
            self.train_mat = self.train_mat[0, :]
            self.train_mat = self.train_mat.reshape((1, len(self.train_mat)))
            perturbations = self._generate_perturbations(num_perturb)
            if not np.all(self.train_mat_sel_idx == 1):
                perturbations[:, self.train_mat_sel_idx.astype(bool)] = 1  # no perturbation for col idx=1
                                                                        # 0 mean selected indices
            else:
                break

            self._populate_train_matrix(perturbations, True)
            train_matrix = self.train_mat
            X = train_matrix[:, :-1]
            y = train_matrix[:, -1]
            X = X[:, np.logical_not(self.train_mat_sel_idx.astype(bool))]  # selecting only columns that have 0
            original_image = np.ones(X.shape[1])[np.newaxis, :]
            distances = sklearn.metrics.pairwise_distances(X, original_image, metric='cosine').ravel()
            distances.shape
            kernel_width = 0.25
            weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function
            coeffs_bs = []
            for i in range(n_iterations):
                indices_bs = random.choices(range(len(X)), weights=weights, k=len(X))
                X_sample, y_sample = X[indices_bs], y[indices_bs]
                weights_bs = weights[indices_bs]
                # X_sample, y_sample = resample(X, y, replace=True, n_samples=self.train_mat.shape[0])
                ridge_model = Ridge(alpha=1)
                ridge_model.fit(X_sample, y_sample, sample_weight=weights_bs)
                coeffs_bs.append(ridge_model.coef_)
            coeffs_bs = np.array(coeffs_bs)

            sign_entropies = []
            for column in range(coeffs_bs.shape[1]):
                data = coeffs_bs[:, column]
                sign_entropy = self._calculate_entropy(data)
                sign_entropies.append(sign_entropy)

            num_predictors = len(sign_entropies)
            sign_entropies = np.array(sign_entropies)
            av_sign_entropy = np.mean(sign_entropies)
            ratio_zero_entropy = np.count_nonzero(sign_entropies == 0) / (sign_entropies.shape[0])

            non_zero_indices = np.where(sign_entropies != 0)[0]
            zero_ent_indices = np.where(sign_entropies == 0)[0]

            print("******Non Zero Indices: ", str(non_zero_indices))
            # mapping the non_zero_indices to the original feature set
            original_0_indices = np.where(self.train_mat_sel_idx == 0)[0]
            mapped_non0_indices = original_0_indices[non_zero_indices]

            if not np.size(mapped_non0_indices) == 0:
                self.train_mat_sel_idx[mapped_non0_indices] = 1
                tolerance_cur = 0
            else:
                if tolerance_cur == 0:
                    final_coeffs = coeffs_bs
                    tolerance_cur = tolerance_cur + 1
                else:
                    if tolerance_cur < tolerance_limit:
                        tolerance_cur = tolerance_cur + 1  # re-evaluating n times after a good run
                    else:
                        break  # terminate the feature elimination process

        if rank_stabilization == True:
            final_coeffs = pd.DataFrame(final_coeffs)

            try:
                final_coeffs.columns = np.nonzero(self.train_mat_sel_idx == 0)[0]
            except ValueError:
                print("Error when assigning columns!")
                print("final_coeffs shape:", final_coeffs.shape)
                print("Non-zero indices length:", len(np.nonzero(self.train_mat_sel_idx == 0)[0]))
                print("self.train_mat_sel_idx:", self.train_mat_sel_idx)
                print("Non-zero indices:", np.nonzero(self.train_mat_sel_idx == 0)[0])

            positive_cols = final_coeffs.columns[(final_coeffs > 0).all()]
            negative_cols = final_coeffs.columns[(final_coeffs < 0).all()]

            # DataFrames with only positive and negative columns
            positive_df = final_coeffs[positive_cols]
            negative_df = final_coeffs[negative_cols]

            column_means = positive_df.mean()
            column_names = positive_df.columns.to_numpy()
            pos_dict = {
                'column_names': column_names,
                'column_means': column_means.to_numpy()
            }

            column_means_neg = negative_df.mean()
            column_names_neg = negative_df.columns.to_numpy()
            neg_dict = {
                'column_names': column_names_neg,
                'column_means': column_means_neg.to_numpy()
            }

            negative_df = negative_df.abs() # just sort the negative coeffs by magnitude

            pos_feature_ranks = self._get_average_ranks(positive_df)
            neg_feature_ranks = self._get_average_ranks(negative_df)
        else:
            X = self.train_mat[:,0:(self.train_mat.shape[1] - 1)]
            retained_indices = np.where(~self.train_mat_sel_idx.astype(bool))[0]
            if len(retained_indices)>0 and \
                    np.var(self.train_mat[:, (self.train_mat.shape[1] - 1):self.train_mat.shape[1]])!=0:
                X = X[:, retained_indices]
                y = self.train_mat[:, (self.train_mat.shape[1] - 1):self.train_mat.shape[1]]
                original_image = np.ones(X.shape[1])[np.newaxis, :]
                distances = sklearn.metrics.pairwise_distances(X, original_image, metric='cosine').ravel()
                kernel_width = 0.25
                weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function
                ridge_model = Ridge(alpha=1)
                ridge_model.fit(X, y, sample_weight=weights)
                mapped_coefficients = np.zeros(self.train_mat.shape[1] - 1)
                mapped_coefficients[retained_indices] = ridge_model.coef_

                # Get the original indices for positive and negative coefficients
                positive_coef_indices = np.where(mapped_coefficients > 0)[0]
                negative_coef_indices = np.where(mapped_coefficients < 0)[0]

                # Get the coefficients using the indices
                positive_coefs = mapped_coefficients[positive_coef_indices]
                negative_coefs = mapped_coefficients[negative_coef_indices]

                # Get the indices that would sort the coefficients
                positive_coefs_sorted_indices = positive_coef_indices[np.argsort(positive_coefs)[::-1]] #np.argsort(positive_coefs)[::-1]  # Descending order
                negative_coefs_sorted_indices = negative_coef_indices[np.argsort(negative_coefs)]  # Ascending order

                positive_coefs_sorted = positive_coefs[np.argsort(positive_coefs)[::-1]]
                negative_coefs_sorted = negative_coefs[np.argsort(negative_coefs)]
                pos_dict = {
                    'column_names': positive_coefs_sorted_indices,
                    'column_means': positive_coefs_sorted
                }

                neg_dict = {
                    'column_names': negative_coefs_sorted_indices,
                    'column_means': negative_coefs_sorted
                }
            else:
                positive_coefs_sorted_indices, negative_coefs_sorted_indices,\
                pos_dict, neg_dict = self.get_lime_explanations(num_perturb=num_perturb, sigma=sigma)

            pos_feature_ranks = positive_coefs_sorted_indices
            neg_feature_ranks = negative_coefs_sorted_indices

        return np.nonzero(self.train_mat_sel_idx)[0], pos_feature_ranks, neg_feature_ranks, \
               step*num_perturb, pos_dict, neg_dict

    def _get_average_ranks(self, df):
        # df = pd.DataFrame(df)
        df = df.rank(axis=1, method='min')

        # Number of features (candidates in Borda count)
        n_candidates = df.shape[1]

        df_borda = df.rank(axis=1, ascending=False)

        # Sum the Borda counts for each feature
        borda_count = df_borda.sum()

        # Display the result
        borda_count = borda_count.sort_values(ascending=False)

        return np.array(list(borda_count.items()))

    def get_lime_explanations(self, num_perturb=500, sigma=0):
        self.sigma=sigma
        self.train_mat = self.train_mat[0, :]
        self.train_mat = self.train_mat.reshape((1, len(self.train_mat)))
        perturbations = self._generate_perturbations(num_perturb)
        self._populate_train_matrix(perturbations, True)
        train_matrix = self.train_mat
        X = train_matrix[:, :-1]
        y = train_matrix[:, -1]
        original_image = np.ones(X.shape[1])[np.newaxis, :]
        distances = sklearn.metrics.pairwise_distances(X, original_image, metric='cosine').ravel()
        distances.shape
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))  # Kernel function
        ridge_model = Ridge(alpha=1)
        ridge_model.fit(X, y, sample_weight=weights)
        coefficients = ridge_model.coef_

        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_coefficients = coefficients[sorted_indices]

        # Partition coefficients into positive and negative
        positive_coefficients = sorted_coefficients[sorted_coefficients >= 0]
        negative_coefficients = sorted_coefficients[sorted_coefficients < 0]

        pos_indices = sorted_indices[sorted_coefficients >= 0]
        neg_indices = sorted_indices[sorted_coefficients < 0]
        # Create dictionaries
        pos_dict = {
            'column_names': pos_indices,
            'column_means': positive_coefficients
        }

        neg_dict = {
            'column_names': neg_indices,
            'column_means': negative_coefficients
        }

        return pos_indices, neg_indices, pos_dict, neg_dict
