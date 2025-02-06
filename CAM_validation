#Load CAM++, Guided_CAM, or HiResCAM
map_path = "cam++.nii"  
saliency_map = nib.load(map_path).get_fdata()
#Load the model
loaded_model = keras.models.load_model("../LeNet_layer6_v1.keras")


# Calculating faithfulness score using occlusion method
import numpy as np

def calculate_faithfulness(model, image, saliency_map, occlusion_percentage=0.1):
    """
    Calculate faithfulness by occluding regions in the image based on the saliency map.

    :param model: The trained model (with a predict() function)
    :param image: Original input image (shape: [128, 128, 50] for 3D)
    :param saliency_map: Saliency map (same shape as the image)
    :param occlusion_percentage: Percentage of most important pixels to occlude.

    :return: Drop in predicted probability as a faithfulness score.
    """
    original_pred = model.predict(np.expand_dims(image, axis=0))[0]

    # Flatten the saliency map and find the most important pixels
    flattened_saliency = saliency_map.flatten()
    num_pixels = int(flattened_saliency.size * occlusion_percentage)

    # Get the indices of the top important pixels
    top_pixels_indices = np.argsort(flattened_saliency)[-num_pixels:]

    # Create a copy of the original image and flatten it for modification
    image_flattened = image.flatten()

    # Occlude the most important pixels (e.g., set them to 0-->Blackout method)
    image_flattened[top_pixels_indices] = 0

    # Reshape back to the original image shape
    occluded_image = image_flattened.reshape(image.shape)
    print(occluded_image.shape)
    plt.imshow(occluded_image[:,:,30], cmap='gray')
    #plt.imshow(img_arr[:,:,30], cmap='gray')
    plt.axis('off')
    plt.show()

    # Measure the new prediction after occlusion
    occluded_pred = model.predict(np.expand_dims(occluded_image, axis=0))[0]
    print("Original", original_pred)
    print("Occluded", occluded_pred)

    # Faithfulness score: drop in prediction probability
    faithfulness_score = (original_pred - occluded_pred)
    return faithfulness_score
print(img_arr.shape)
faithscore=calculate_faithfulness(loaded_model, img_arr, saliency_map)
print("Faithfulness_score",faithscore)


#Finding iAUC through progressive masking
#calculating iAUC for CAM++, Guided_CAM nd HiResCAM

# Function to progressively mask the input based on saliency scores
def progressive_masking(input_image, saliency_map, model, steps=100):
    """
    Mask the input image based on saliency map progressively and calculate model outputs.

    Args:
        input_image (numpy array): Original input (3D for medical images).
        saliency_map (numpy array): Saliency map corresponding to the input.
        model (tf.keras.Model): Trained CNN model.
        steps (int): Number of steps for masking.

    Returns:
        np.array: Model predictions at each step.
    """
    sorted_indices = np.argsort(-saliency_map.flatten())  # Sort in descending order
    flat_image = input_image.flatten()

    # Store model outputs as we progressively mask the input
    outputs = []

    for i in range(steps + 1):
        # Calculate the number of pixels to mask
        num_pixels_to_mask = int((i / steps) * flat_image.size)

        # Create a copy of the original image and apply masking
        masked_image = flat_image.copy()
        masked_image[sorted_indices[:num_pixels_to_mask]] = 0  # Mask top important regions

        # Reshape to original dimensions and get model output
        masked_image = masked_image.reshape(input_image.shape)
        prediction = model.predict(masked_image[np.newaxis, ...])

        # Append the prediction for this masking step
        outputs.append(prediction)
    #plt.imshow(masked_image[:,:,30], cmap='gray')
    #plt.axis('off')
    return np.array(outputs)

# Function to compute iAUC
def compute_iAUC(predictions, steps):
    """
    Compute the integral area under the curve (iAUC).

    Args:
        predictions (numpy array): Model predictions at each masking step.
        steps (int): Number of steps for masking.

    Returns:
        float: iAUC value.
    """
    # Normalize x-axis by steps
    x_values = np.linspace(0, 1, steps + 1)

    # Reshape predictions to be 1D for compatibility with x_values
    predictions = predictions[:, 0, 0] # This line is changed to extract the relevant data

    # Plot curve between fraction of image masked (x) and model output (y)
    auc = np.trapz(predictions, x_values)  # AUC using trapezoidal rule
    plt.plot(x_values, predictions)
    return auc

# Step 1: Progressive masking
predictions = progressive_masking(img_arr, saliency_map, loaded_model, steps=100)

# Step 2: Compute iAUC
iAUC_value = compute_iAUC(predictions, steps=100)
print(f"iAUC value: {iAUC_value}")
