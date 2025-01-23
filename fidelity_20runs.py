import gc
import cv2
import keras
import copy
from skimage import filters
import os
import pickle
from scipy.stats import pearsonr
import tensorflow as tf
from sklearn.metrics import auc
from PIL import Image
import numpy as np


resnet_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
inceptionv3_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')

img_info_filepath = "img_info_dict/imgs_info.pkl"
with open(img_info_filepath, 'rb') as f:
    img_info_dict = pickle.load(f)


def save_image(image_data, save_path):
    """
    Save an image to disk, handling both PIL.Image objects and NumPy arrays.

    Args:
    - image_data: The image to save, either a PIL.Image object or a NumPy array.
    - save_path (str): The path where the image should be saved.
    """
    # If it's a NumPy array, convert to uint8 and then to a PIL Image
    if isinstance(image_data, np.ndarray):
        if image_data.dtype != np.uint8:
            image_data = (image_data * 255).astype(np.uint8) if image_data.max() <= 1 else image_data.astype(np.uint8)
        img = Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        img = image_data
    else:
        raise TypeError("Input must be a PIL.Image object or a NumPy array.")

    # Save the image
    img.save(save_path)
    print(f"Image saved at {save_path}")



def perturb_image(img, perturbation, segments):
    """
    Perturb the given image based on the perturbation vector and segments.

    Parameters:
    - img (array): Original image.
    - perturbation (array): Binary vector indicating which segments to perturb. 0 for perturbation and
    1 for no perturbation
    - segments (array): Segments/superpixels of the image.

    Returns:
    - perturbed_image (array): Perturbed version of the original image.
    """

    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, np.newaxis]
    return perturbed_image

def perturb_image_blur(img, perturbation, segments, sigma=1):
    """
    Perturb the given image with Gaussian blur based on the perturbation vector and segments.

    Parameters:
    - img (array): Original image.
    - perturbation (array): Binary vector indicating which segments to perturb.
    - segments (array): Segments/superpixels of the image.
    - sigma (float, optional): Sigma value for the Gaussian blur. Default is 1.

    Returns:
    - perturbed_image (array): Blurred version of the original image.
    """

    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1
    mask3d = cv2.merge((mask, mask, mask))
    perturbed_image = copy.deepcopy(img)
    blur_img = filters.gaussian(img, sigma)
    perturbed_image = np.where(mask3d == np.array([0.0, 0.0, 0.0]), blur_img, img)
    return perturbed_image


def compute_aopc_del(image_path, segments, positive_ranks, negative_ranks, model_name, sigma, po, num_ranks=5):
    """
    Compute the AOPC metric for an image, given its segments and positive ranks.

    Args:
    - image (np.ndarray): Original image array of shape (H, W, 3).
    - segments (np.ndarray): Array of segments of shape (H, W), where each unique value represents a segment.
    - positive_ranks (list): List of segments' ranks, with most positive (important) segment first.

    Returns:
    - aopc (float): Computed AOPC value.
    """

    # Load pretrained ResNet50 model with ImageNet weights
    if model_name == "resnet50":
        model = resnet_model
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        decode_predictions = tf.keras.applications.resnet.decode_predictions
        target_img_size = (224,224)
    else:
        if model_name == "inceptionv3":
            model = inceptionv3_model
            preprocess_input = keras.applications.inception_v3.preprocess_input
            decode_predictions = keras.applications.inception_v3.decode_predictions
            target_img_size = (299, 299)
        else:
            print("Error")
            return -3

    # Preprocess the image and get its prediction
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img_arr = preprocess_input(img_arr)
    pred0 = model.predict(np.array([img_arr]), verbose=0)
    top_pred_class = pred0[0].argsort()[-1:][::-1]
    prob0 = pred0[0][top_pred_class]
    #print(pred0)
    # L is the length of positive ranks
    if (positive_ranks is None and negative_ranks is None): # or len(positive_ranks) < num_ranks \
            #or len(negative_ranks) < num_ranks:
        return float('nan')

    #positive features
    if num_ranks == -1:
        num_pos_ranks = len(positive_ranks)
    else:
        num_pos_ranks = num_ranks

    # Initialize sum
    aopcs_pos = []
    if po == 'L':
        positive_ranks = np.flip(positive_ranks)

    for k in range(0, num_pos_ranks):
        #print(positive_ranks[k])
        perturbation = np.ones(len(np.unique(segments)))
        cur_ranks = np.array(positive_ranks[0:k+1], dtype=int)
        perturbation[cur_ranks] = 0
        img_arr = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(img))
        pert_img = perturb_image_blur(img_arr,perturbation=perturbation, segments=segments, sigma=sigma)
        pert_img = tf.keras.preprocessing.image.img_to_array(pert_img)
        pert_img_arr = preprocess_input(pert_img)
        pred1 = model.predict(np.array([pert_img_arr]), verbose=0)
        prob1 = pred1[0][top_pred_class]
        #print(prob1)
        prob_diff = prob0[0]-prob1[0]
        aopcs_pos.append(prob_diff)
        del img_arr, pert_img, pert_img_arr

    # negative features
    if num_ranks == -1:
        num_neg_ranks = len(negative_ranks)
    else:
        num_neg_ranks = num_ranks

    # Initialize sum
    aopcs_neg = []
    if po == 'L':
        negative_ranks = np.flip(negative_ranks)

    for k in range(0, num_neg_ranks):
        #print(positive_ranks[k])
        perturbation = np.ones(len(np.unique(segments)))
        cur_ranks = np.array(negative_ranks[0:k+1], dtype=int)
        perturbation[cur_ranks] = 0
        img_arr_tmp = tf.keras.preprocessing.image.img_to_array(img)
        pert_img = perturb_image_blur(img_arr_tmp,perturbation=perturbation, segments=segments, sigma=sigma)
        pert_img_arr = preprocess_input(pert_img)
        pred1 = model.predict(np.array([pert_img_arr]), verbose=0)
        prob1 = pred1[0][top_pred_class]
        #print(prob1)
        prob_diff = prob1[0] - prob0[0]
        aopcs_neg.append(prob_diff)
        del img_arr_tmp, pert_img, pert_img_arr

    if len(aopcs_pos)==0:
        aopcs_pos.append(0)
    if len(aopcs_neg)==0:
        aopcs_neg.append(0)

    return np.mean(np.array(aopcs_pos)), np.mean(np.array(aopcs_neg))


def compute_aopc_ins(image_path, segments, positive_ranks, negative_ranks, model_name, sigma,po, num_ranks=5):
    """
    Compute the AOPC metric for an image, given its segments and positive ranks.

    Args:
    - image (np.ndarray): Original image array of shape (H, W, 3).
    - segments (np.ndarray): Array of segments of shape (H, W), where each unique value represents a segment.
    - positive_ranks (list): List of segments' ranks, with most positive (important) segment first.

    Returns:
    - aopc (float): Computed AOPC value.
    """

    # Load pretrained ResNet50 model with ImageNet weights
    if model_name == "resnet50":
        model = resnet_model
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        decode_predictions = tf.keras.applications.resnet.decode_predictions
        target_img_size = (224,224)
    else:
        if model_name == "inceptionv3":
            model = inceptionv3_model
            preprocess_input = keras.applications.inception_v3.preprocess_input
            decode_predictions = keras.applications.inception_v3.decode_predictions
            target_img_size = (299, 299)
        else:
            print("Error")
            return -3

    # Preprocess the image and get its prediction
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    perturbation = np.zeros(len(np.unique(segments)))
    img_arr = perturb_image_blur(img_arr,perturbation=perturbation, segments=segments, sigma=sigma)
    img_arr = preprocess_input(img_arr)
    pred0 = model.predict(np.array([img_arr]), verbose=0)
    top_pred_class = pred0[0].argsort()[-1:][::-1]
    prob0 = pred0[0][top_pred_class]
    #print(pred0)
    # L is the length of positive ranks
    if (positive_ranks is None and negative_ranks is None): # or len(positive_ranks) < num_ranks \
            #or len(negative_ranks) < num_ranks:
        return float('nan')

    #positive features
    if num_ranks == -1:
        num_pos_ranks = len(positive_ranks)
    else:
        num_pos_ranks = num_ranks

    # Initialize sum
    aopcs_pos = []

    if po == 'L':
        positive_ranks = np.flip(positive_ranks)

    for k in range(0, num_pos_ranks):
        #print(positive_ranks[k])
        perturbation = np.zeros(len(np.unique(segments)))
        cur_ranks = np.array(positive_ranks[0:k+1], dtype=int)
        perturbation[cur_ranks] = 1
        img_arr = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(img))
        pert_img = perturb_image_blur(img_arr,perturbation=perturbation, segments=segments, sigma=sigma)
        pert_img = tf.keras.preprocessing.image.img_to_array(pert_img)
        pert_img_arr = preprocess_input(pert_img)
        pred1 = model.predict(np.array([pert_img_arr]), verbose=0)
        prob1 = pred1[0][top_pred_class]
        #print(prob1)
        prob_diff = prob1[0]-prob0[0]
        aopcs_pos.append(prob_diff)
        del img_arr, pert_img, pert_img_arr

    # negative features
    if num_ranks == -1:
        num_neg_ranks = len(negative_ranks)
    else:
        num_neg_ranks = num_ranks

    if po == 'L':
        negative_ranks = np.flip(negative_ranks)

    # Initialize sum
    aopcs_neg = []
    for k in range(0, num_neg_ranks):
        #print(positive_ranks[k])
        perturbation = np.zeros(len(np.unique(segments)))
        cur_ranks = np.array(negative_ranks[0:k+1], dtype=int)
        perturbation[cur_ranks] = 1
        img_arr_tmp = tf.keras.preprocessing.image.img_to_array(img)
        pert_img = perturb_image_blur(img_arr_tmp,perturbation=perturbation, segments=segments, sigma=sigma)
        pert_img_arr = preprocess_input(pert_img)
        pred1 = model.predict(np.array([pert_img_arr]), verbose=0)
        prob1 = pred1[0][top_pred_class]
        #print(prob1)
        prob_diff = prob0[0] - prob1[0]
        aopcs_neg.append(prob_diff)
        del img_arr_tmp, pert_img, pert_img_arr

    if len(aopcs_pos)==0:
        aopcs_pos.append(0)
    if len(aopcs_neg)==0:
        aopcs_neg.append(0)

    return np.mean(np.array(aopcs_pos)),  np.mean(np.array(aopcs_neg))


def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    if arr.size > 1:
        adjusted_sum = (arr.sum() - arr[0] / 2 - arr[-1] / 2)
        result = adjusted_sum / (arr.shape[0] - 1)
    else:
        # If arr has only one element, result is that element
        result = arr[0] if arr.size == 1 else np.nan

    return result


def compute_ins_auc(image_path, segments, positive_ranks, negative_ranks, model_name, sigma, po, num_ranks=5):
    model, preprocess_input, target_img_size = load_model_and_preprocessing(model_name)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    pred0 = model.predict(np.array([preprocess_input(img_arr)]), verbose=0)
    top_pred_class = np.argmax(pred0[0])
    prob0 = pred0[0][top_pred_class] # this is unperturbed Prob0 and overwritten later.
                                    # So, not used

    # Initial model prediction on the fully blurred image
    perturbation = np.zeros(len(np.unique(segments)))  # No segments are initially inserted
    perturbed_img = perturb_image_blur(tf.keras.preprocessing.image.img_to_array(img), perturbation=perturbation, segments=segments, sigma=sigma)
    perturbed_img_pp = copy.deepcopy(perturbed_img)
    perturbed_img_pp = preprocess_input(perturbed_img_pp)
    pred0 = model.predict(np.array([perturbed_img_pp]), verbose=0)
    #top_pred_class = np.argmax(pred0[0]) # we use top pred class from unperturbed image
    prob0 = pred0[0][top_pred_class]

    # Return NaN if no ranks are provided
    if (positive_ranks is None or len(positive_ranks) == 0) and (negative_ranks is None or len(negative_ranks) == 0):
        return float('nan'), float('nan')

    # Positive feature insertion
    ins_pos_scores = []
    if positive_ranks is not None and len(positive_ranks) > 0:
        if po == 'L':
            positive_ranks = np.flip(positive_ranks)
        #num_pos_ranks = len(positive_ranks) if num_ranks == -1 else num_ranks
        num_pos_ranks = min(len(positive_ranks), num_ranks)
        print(f"pos ranks: {positive_ranks} and segments: {np.unique(segments)}")
        for k in np.arange(0, num_pos_ranks):
            perturbed_img_copy = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(perturbed_img))
            img_arr = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(img))
            # Insert original pixel values for the specified segments
            cur_ranks = np.array(positive_ranks[0:k+1], dtype=int)
            for rank in cur_ranks:
                perturbed_img_copy[segments == rank] = img_arr[segments == rank]

            # Predict with the progressively "de-perturbed" image
            pert_img_arr = preprocess_input(perturbed_img_copy)
            prob1 = model.predict(np.array([pert_img_arr]), verbose=0)[0][top_pred_class]
            ins_pos_scores.append(prob1-prob0)

            del pert_img_arr, perturbed_img_copy, img_arr
            gc.collect()

    # Negative feature insertion
    ins_neg_scores = []
    if negative_ranks is not None and len(negative_ranks) > 0:
        if po == 'L':
            negative_ranks = np.flip(negative_ranks)
        #num_neg_ranks = len(negative_ranks) if num_ranks == -1 else num_ranks
        num_neg_ranks = min(len(negative_ranks), num_ranks)
        print(f"neg ranks: {negative_ranks} and segments: {np.unique(segments)}")

        for k in np.arange(0, num_neg_ranks):
            perturbed_img_copy = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(perturbed_img))
            img_arr = tf.keras.preprocessing.image.img_to_array(img)

            # Insert original pixel values for the specified segments
            cur_ranks = np.array(negative_ranks[0:k+1], dtype=int)
            for rank in cur_ranks:
                perturbed_img_copy[segments == rank] = img_arr[segments == rank]

            # Predict with the progressively "de-perturbed" image
            pert_img_arr = preprocess_input(perturbed_img_copy)
            prob1 = model.predict(np.array([pert_img_arr]), verbose=0)[0][top_pred_class]
            ins_neg_scores.append(prob0 - prob1) # for neg the insertion prob should ideally go down

            del pert_img_arr, perturbed_img_copy
            gc.collect()

    # Calculate AUC for positive scores and normalize by the number of features
    if len(ins_pos_scores) > 1:
        pos_auc = auc(np.linspace(0, 1, len(ins_pos_scores)), ins_pos_scores) / num_pos_ranks
    elif len(ins_pos_scores) == 1:
        pos_auc = (0.5 * (ins_pos_scores[0] - prob0) * 1 + (ins_pos_scores[0] - prob0) * 1) / num_pos_ranks
    else:
        pos_auc = 0.0  # or another default value if no scores are available

        # Calculate AUC for negative scores and normalize by the number of features
    if len(ins_neg_scores) > 1:
        neg_auc = auc(np.linspace(0, 1, len(ins_neg_scores)), ins_neg_scores) / num_neg_ranks
    elif len(ins_neg_scores) == 1:
        neg_auc = (0.5 * (ins_neg_scores[0] - prob0) * 1 + (ins_neg_scores[0] - prob0) * 1) / num_neg_ranks
    else:
        neg_auc = 0.0  # or another default value if no scores are available

    return pos_auc, neg_auc


def compute_del_auc(image_path, segments, positive_ranks, negative_ranks, model_name, sigma, po, num_ranks=5):
    model, preprocess_input, target_img_size = load_model_and_preprocessing(model_name)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    pred0 = model.predict(np.array([preprocess_input(img_arr)]), verbose=0)
    top_pred_class = np.argmax(pred0[0])
    prob0 = pred0[0][top_pred_class]

    # build the blurred image
    blurred_img = filters.gaussian(tf.keras.preprocessing.image.img_to_array(img), sigma=sigma)
    #save_image(img, 'img.jpg') # uncomment if you want to see the intermediate blurred images
    #save_image(blurred_img,'blurred_img.jpg')

    # Return NaN if no ranks are provided
    if (positive_ranks is None or len(positive_ranks) == 0) and (negative_ranks is None or len(negative_ranks) == 0):
        return float('nan'), float('nan')

    num_pos_ranks = min(len(positive_ranks), num_ranks)
    num_neg_ranks = min(len(negative_ranks), num_ranks)

    # Positive feature deletion
    del_pos_scores = [prob0] # inserting the unperturbed probability of top class
    if positive_ranks is not None and len(positive_ranks) > 0:
        if po == 'L':
            positive_ranks = np.flip(positive_ranks)
        print(f"pos ranks: {positive_ranks} and segments: {np.unique(segments)}")

        for k in np.arange(0, num_pos_ranks):
            perturbed_img = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(img))
            cur_ranks = np.array(positive_ranks[0:k + 1], dtype=int)

            # Apply blur to the specified segments
            for rank in cur_ranks:
                perturbed_img[segments == rank] = blurred_img[segments == rank]

            # Predict with the progressively blurred image
            pert_img_arr = preprocess_input(perturbed_img)
            prob1 = model.predict(np.array([pert_img_arr]), verbose=0)[0][top_pred_class]
            prob_diff = -1*(prob0 - prob1)  ## using - sign to have lower auc for explanations
                                            ## for which the prob drops
            del_pos_scores.append(prob1)

            del pert_img_arr, perturbed_img
            gc.collect()

    # Negative feature deletion
    del_neg_scores = [prob0]
    if negative_ranks is not None and len(negative_ranks) > 0:
        if po == 'L':
            negative_ranks = np.flip(negative_ranks)
        print(f"neg ranks: {negative_ranks} and segments: {np.unique(segments)}")

        for k in np.arange(0, num_neg_ranks):
            perturbed_img = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(img))
            cur_ranks = np.array(negative_ranks[0:k + 1], dtype=int)

            # Apply blur to the specified segments
            for rank in cur_ranks:
                perturbed_img[segments == rank] = blurred_img[segments == rank]

            # Predict with the progressively blurred image
            pert_img_arr = preprocess_input(perturbed_img)
            prob1 = model.predict(np.array([pert_img_arr]), verbose=0)[0][top_pred_class]
            prob_diff = -1*(prob1 - prob0)  ## using - sign to have -ve auc for explanations with
                                            ## worse fidelity

            del_neg_scores.append(prob_diff)

            del pert_img_arr, perturbed_img
            gc.collect()

    # Calculate AUC for positive and negative scores
    if len(del_pos_scores) > 1:
        pos_auc = auc(np.linspace(0, 1, len(del_pos_scores)), del_pos_scores) / num_pos_ranks
    elif len(del_pos_scores) == 1:
        pos_auc = (0.5 * (del_pos_scores[0] - prob0) * 1 + (del_pos_scores[0] - prob0) * 1) / num_pos_ranks
    else:
        pos_auc = 0.0

    if len(del_neg_scores) > 1:
        neg_auc = auc(np.linspace(0, 1, len(del_neg_scores)), del_neg_scores) / num_neg_ranks
    elif len(del_neg_scores) == 1:
        neg_auc = (0.5 * (del_neg_scores[0] - prob0) * 1 + (del_neg_scores[0] - prob0) * 1) / num_neg_ranks
    else:
        neg_auc = 0.0

    return pos_auc, neg_auc


def compute_faithfulness(image_path, segments, positive_ranks, negative_ranks, model_name, sigma, po, num_ranks=5):
    model, preprocess_input, target_img_size = load_model_and_preprocessing(model_name)
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_img_size)
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    pred0 = model.predict(np.array([preprocess_input(img_arr)]), verbose=0)
    top_pred_class = np.argmax(pred0[0])
    prob0 = pred0[0][top_pred_class]

    # Initial model prediction on the fully blurred image
    blurred_img = filters.gaussian(tf.keras.preprocessing.image.img_to_array(img), sigma=10)

    def get_ranked_correlations(ranks, is_positive=True):
        """Calculate Pearson correlation for ranks with drop in probability."""
        if ranks is None:
            return np.nan

        drop_scores = []
        inv_ranks = []  # Inverted ranks: smaller ranks = higher relevance
        num_ranks_actual = min(len(ranks), num_ranks)
        for k in np.arange(0, num_ranks_actual, step=1):
            perturbed_img = tf.keras.preprocessing.image.img_to_array(copy.deepcopy(img))

            perturbed_img[segments == ranks[k]] = blurred_img[segments == ranks[k]]

            pert_img_arr = preprocess_input(perturbed_img)
            prob1 = model.predict(np.array([pert_img_arr]), verbose=0)[0][top_pred_class]

            if is_positive== True:
                prob_diff = prob0 - prob1 # Drop in probability from baseline for pos features
            else:
                prob_diff = prob1 - prob0 # Lift in probability from baseline for neg features

            drop_scores.append(prob_diff)

            inv_ranks.append(num_ranks_actual - k)  # Invert ranks: higher relevance has lower rank number

            del pert_img_arr, perturbed_img
            gc.collect()

        # Calculate Pearson correlation of inverted ranks with drop scores
        correlation, _ = pearsonr(inv_ranks, drop_scores) if len(drop_scores) > 1 else (np.nan, np.nan)
        return correlation

    # Calculate Faithfulness scores
    pos_faithfulness = get_ranked_correlations(positive_ranks, is_positive=True)
    neg_faithfulness = get_ranked_correlations(negative_ranks, is_positive=False)

    return pos_faithfulness, neg_faithfulness


def load_model_and_preprocessing(model_name):
    """
    Load the model and the associated preprocessing based on the model name.
    """
    if model_name == "resnet50":
        model = resnet_model
        preprocess_input = tf.keras.applications.resnet.preprocess_input
        target_img_size = (224,224)
    elif model_name == "inceptionv3":
        model = inceptionv3_model
        preprocess_input = keras.applications.inception_v3.preprocess_input
        target_img_size = (299, 299)
    else:
        raise ValueError("Unsupported model name")
    return model, preprocess_input, target_img_size


# Define the necessary models
resnet_model = tf.keras.applications.resnet50.ResNet50(weights='imagenet')
inceptionv3_model = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet')

# Load image information
img_info_filepath = "img_info_dict/imgs_info.pkl"
with open(img_info_filepath, 'rb') as f:
    img_info_dict = pickle.load(f)

#Function mapping for fidelity metrics
fidelity_functions = {
    "aopc_ins": compute_aopc_ins,
    "aopc_del": compute_aopc_del,
    "ins": compute_ins_auc,
    "del": compute_del_auc
}


# Aggregate function, adjusted to accept a metric function as a parameter
def aggregate_score(segments, result_file_path, image_path, model_name, sigma, po, metric_func):
    aopc_scores = []
    with open(result_file_path, 'rb') as f:
        data = pickle.load(f)
        imgs = data.keys()

        for img in imgs:
            run_dict = data[img][0]
            runs = list(run_dict.keys()) # test TODO remove for full run
            #runs = runs[0:1] # test TODO remove for full run
            for run in runs:
                run_info = run_dict[run]
                selected_key_pos = 'pos' if 'pos' in run_info[0] else 'pos_borda' if 'pos_borda' in run_info[0] else None
                selected_key_neg = 'neg' if 'neg' in run_info[0] else 'neg_borda' if 'neg_borda' in run_info[0] else None

                pos_ranks = run_info[0][selected_key_pos]
                neg_ranks = run_info[0][selected_key_neg]

                # Use the provided metric function
                pos_score, neg_score = metric_func(
                    image_path, segments, positive_ranks=pos_ranks, negative_ranks=neg_ranks,
                    model_name=model_name, sigma=sigma, po=po
                )
                aopc_scores.append({"pos": pos_score, "neg": neg_score})

    return aopc_scores

# Directories and parameters
dataset_dir = "results/" # this is the results dir after running slice_test.py
datasets = os.listdir(dataset_dir)
img_src_dir = "images_all/" # directory of all images used in the experiment
fidelity_res_dir = "fidelity_results/" # create a new directory to store the fidelity results. It should have aopc_ins, aopc_del, ins and del sub-directories for all the 4 metrics
po = 'M' # most relevant first or least relevant first. Right now this code handles only most relevant first

# Loop over datasets and metrics
for metric_name, metric_func in fidelity_functions.items():
    for dataset_name in datasets:
        print(f"metric name: {metric_name} and dataset: {dataset_name}")
        dirs = os.listdir(dataset_dir + dataset_name)
        for dir in dirs:
            # Define directory and filename based on the metric
            metric_dir = os.path.join(fidelity_res_dir, metric_name)
            os.makedirs(metric_dir, exist_ok=True)
            pkl_filename = os.path.join(metric_dir, f"{dataset_name}_{dir}_{metric_name}_all_features_{po}.pkl")

            if os.path.exists(pkl_filename):
                print(f"Exists: {pkl_filename}")
                continue
            else:
                dataset_dict = {}
                model_name = dir.split("_")[1]
                xai_name = dir.split("_")[0]

                res_files = os.listdir(dataset_dir + dataset_name + "/" + dir)
                #res_files = res_files[0:1]
                for res_file in res_files:
                    img_info_key = model_name + "_" + \
                                   res_file.replace(dataset_name + "_", "").replace(xai_name + "_", "").replace(".jpg", "").replace("_" + model_name + ".pkl", "")

                    try:
                        img_info = img_info_dict[img_info_key][0]
                    except KeyError:
                        print(f"img_info_key Error: {img_info_key}")
                        continue

                    segments = img_info['segments']
                    sel_sigma = img_info['sel_sigma']
                    img_name = res_file.replace(f"{dataset_name}_{xai_name}_", "").replace(f"_{model_name}.pkl", "")
                    img_name += ".jpg" if not img_name.lower().endswith(".jpg") else ""
                    img_path = os.path.join(img_src_dir, img_name)
                    res_file_path = os.path.join(dataset_dir, dataset_name, dir, res_file)

                    # Compute scores using the specified metric function
                    metric_scores = aggregate_score(segments, res_file_path, img_path, model_name, sigma=sel_sigma, po=po, metric_func=metric_func)

                    key = f"{dataset_name}_{model_name}_{xai_name}_{img_name}"
                    dataset_dict[key] = metric_scores

                with open(pkl_filename, 'wb') as f1:
                    pickle.dump(dataset_dict, f1)

