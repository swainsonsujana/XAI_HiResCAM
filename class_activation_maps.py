#Functions to normalize and upsample the 3D gradient maps
def normalize_map(volume):
    """Normalize the volume"""
    min = 0
    max = 1.5
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    #volume = (volume/255.0)*1.5
    return volume
def resize_map(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 60
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    #img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img
# Load the model
# print("Loaded model from disk")
loaded_model = keras.models.load_model("..../LeNet_layer6_v1.keras")

#Load an autistic image from dataset
img=nib.load(img_path)
img_arr=img.get_fdata()
img_arr=resize_volume(img_arr)
img_arr=normalize(img_arr)

#Check the prediction
img_arr1 = img_arr.reshape(1, 128, 128, 60, 1)
#print(background.shape)
pred=loaded_model.predict(img_arr1)
print(pred)
#display the image
plt.imshow(img_arr[:,:,30], cmap='gray')
plt.axis('off')


# Gradient map
with tf.GradientTape() as tape:
    img_arr1_tensor = tf.convert_to_tensor(img_arr1, dtype=tf.float32)  
    tape.watch(img_arr1_tensor)
    result = loaded_model(img_arr1_tensor)
    target_class_index = 0  # Example: Calculate gradient for class 0
    target_class_output = result[0, target_class_index]  # Output for the target class
    grads = tape.gradient(target_class_output, img_arr1_tensor)  # Calculate gradients
print(grads.shape)
grads = np.squeeze(grads)
print(grads.shape)
grads= tf.nn.relu(grads)
grads=grads.numpy()
print(grads.shape)
grads=normalize_map(grads)
grads_rz=resize_map(grads)

# Visualize
plt.imshow(img_arr[:,:,30], cmap='gray')
plt.axis('off')
plt.imshow(grads_rz[:,:,30], cmap='jet', alpha=0.5)
plt.show()

# Save Gradient map
# Create a NIfTI image object from the 3D saliency map array
nifti_img = nib.Nifti1Image(grads_rz, affine=np.eye(4))  # Identity matrix as affine

# Save the image as a NIfTI file
nib.save(nifti_img, '/cam.nii')

# CAM++
!pip install tf-keras-vis
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.scores import CategoricalScore

predicted_class = np.argmax(pred[0])
# Create a GradCAM++ object
gradcam_plus_plus = GradcamPlusPlus(loaded_model, clone=True)
# Define the score function to compute gradients for the predicted class
score = CategoricalScore(predicted_class)
# Compute GradCAM++ heatmap
cam_map = gradcam_plus_plus(score, img_arr1)

# normalize and upsample the CAM++ 
cam_map=tf.nn.relu(cam_map)
cam_map=cam_map.numpy()
cam_map=normalize_map(cam_map)
cam_map=np.squeeze(cam_map)
cam_map=resize_map(cam_map)

# Visualize CAM++
print(cam_map.shape)
plt.imshow(img_arr[:,:,30], cmap='gray')
plt.axis('off')
plt.imshow(cam_map[:,:,30], cmap='jet', alpha=0.5)  
plt.show()
#Save CAM++
# Create a NIfTI image object from the 3D saliency map array
nifti_img = nib.Nifti1Image(cam_map, affine=np.eye(4))  # Identity matrix as affine

# Save the image as a NIfTI file
nib.save(nifti_img, '/camplusplus.nii')


# Guided Grad_CAM
with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(img_arr[np.newaxis, ...])
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    top_pred_index = tf.argmax(preds[0])
    top_class_channel = preds[:, top_pred_index]

grads = tape.gradient(top_class_channel, last_conv_layer_output)[0]
grads=tf.nn.relu(grads)
last_conv_layer_output = last_conv_layer_output[0]

guided_grads = (
    tf.cast(last_conv_layer_output > 0, "float32")
    * tf.cast(grads > 0, "float32")
    * grads
)
pooled_guided_grads = tf.reduce_mean(guided_grads, axis=(0, 1, 2))
guided_gradcam = np.ones(last_conv_layer_output.shape[:2], dtype=np.float32)

# Initialize an empty guided Grad-CAM map with the shape of the 3D spatial dimensions
guided_gradcam = np.ones(last_conv_layer_output.shape[:3], dtype=np.float32)  # Shape: (height, width, depth)

# Create the Guided_CAM heatmap
for channel in range(pooled_guided_grads.shape[0]):
    guided_gradcam += pooled_guided_grads[channel] * last_conv_layer_output[:, :, :, channel]

# Normalize the guided Grad-CAM heatmap
guided_gradcam = tf.squeeze(guided_gradcam).numpy()
guided_gradcam1 = np.clip(guided_gradcam, 0, np.max(guided_gradcam))  # Clip values between 0 and max
guided_gradcam1 = (guided_gradcam1 - guided_gradcam1.min()) / (guided_gradcam1.max() - guided_gradcam1.min())  # Normalize to [0, 1]

print(guided_gradcam1.shape)

guided_gradcam1=resize_map(guided_gradcam1)
guided_gradcam1=np.squeeze(guided_gradcam1)
guided_gradcam1=normalize_map(guided_gradcam1)
guided_gradcam1=guided_gradcam1.astype(np.float32)

#Visualize Guided_CAM
plt.imshow(img_arr[:,:,30], cmap='gray')
plt.axis('off')
plt.imshow(guided_gradcam1[:,:,30], cmap='jet', alpha=0.5)
plt.show()

#Save Guided_CAM
# Create a NIfTI image object from the 3D saliency map array
nifti_img = nib.Nifti1Image(guided_gradcam1, affine=np.eye(4))  # Identity matrix as affine

# Save the image as a NIfTI file
nib.save(nifti_img, '/guided_CAM.nii')

#Guided backpropagation-based Grad CAM with high resolution maps (HiResCAM)
@tf.custom_gradient
def guided_relu(x):
    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad

class GuidedBackprop:
    def __init__(self, model, layer_name: str):
        self.model = model
        self.layer_name = layer_name
        self.gb_model = self.build_guided_model()

    def build_guided_model(self):
        gb_model = tf.keras.Model(
            self.model.inputs, self.model.get_layer(self.layer_name).output
        )
        layers = [
            layer for layer in gb_model.layers[1:] if hasattr(layer, "activation")
        ]
        for layer in layers:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guided_relu
        return gb_model

    def guided_backprop(self, image: np.ndarray):
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            tape.watch(inputs)
            outputs = self.gb_model(inputs)
        grads = tape.gradient(outputs, inputs)[0]
        return grads
gb = GuidedBackprop(loaded_model, "conv3d")

saliency_map = gb.guided_backprop(img_arr[np.newaxis, ...]).numpy()
saliency_map = np.clip(saliency_map, 0, np.max(saliency_map))
saliency_map= saliency_map * grads_rz
saliency_map = np.clip(saliency_map, 0, np.max(saliency_map))
print(saliency_map.shape)

#Aligning the saliency map
saliency_map -= saliency_map.mean()
saliency_map /= saliency_map.std() + tf.keras.backend.epsilon()
saliency_map *= 0.25
saliency_map += 0.5
saliency_map = np.clip(saliency_map, 0, 1)
saliency_map *= (2 ** 8) - 1
saliency_map = saliency_map.astype(np.float32)
saliency_map = np.clip(saliency_map, 0, np.max(saliency_map))

#Visulaize the HiResCAM at slice 35
plt.imshow(img_arr[:,:,35], cmap="gray")
plt.axis('off')
plt.imshow(saliency_map[:,:,35], cmap="jet", alpha=0.5)

#Save HiResCAM
# Create a NIfTI image object from the 3D saliency map array
nifti_img = nib.Nifti1Image(saliencymap, affine=np.eye(4))  # Identity matrix as affine

# Save the image as a NIfTI file
nib.save(nifti_img, '/hirescam.nii')
