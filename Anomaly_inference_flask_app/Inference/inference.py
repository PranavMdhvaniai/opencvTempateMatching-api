from Inference.patchcore.inference import get_patchcore_instance
import pickle
import cv2
import PIL
import os
import logging
from torchvision import transforms
import numpy as np
import torch
from sklearn.metrics import roc_curve

# Configure the logger
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# Create a file handler to write logs to a file
log_file = "inference.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Create a formatter and add it to the file handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
LOGGER.addHandler(file_handler)

class Patchcore:
    """
    A class for handling PatchCore anomaly detection inference.

    Attributes:
        device (torch.device): The device for computation (CPU or GPU).
        patchcore_model: The loaded PatchCore model.
        nn_method: The nearest neighbor method used in the model.
        model_params: Parameters for the loaded model.
        transform_img: Image transformation pipeline.
        inference_batchsize (int): Batch size for inference.
    """
    
    def __init__(self):
        # Automatically detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.faiss_on_gpu = True
            LOGGER.info(f"Using device: {self.device} (GPU)")
        else:
            self.device = torch.device("cpu")
            self.faiss_on_gpu = False
            LOGGER.info(f"Using device: {self.device} (CPU)")
            
        self.patchcore_model = None
        self.nn_method = None   
        self.model_params = None
        self.IMAGENET_MEAN = [0.485, 0.456, 0.406]
        self.IMAGENET_STD = [0.229, 0.224, 0.225]
        self.transform_img = None

    def load_model(self, weights_path):
        """
        Load the PatchCore model and its parameters.

        Args:
            weights_path (str): Path to the model weights directory.
        """
        if self.nn_method is not None:
            self.nn_method.reset_index()

        try:
            self.patchcore_model, self.nn_method = get_patchcore_instance(self.device, weights_path,faiss_on_gpu=self.faiss_on_gpu)
            with open(os.path.join(weights_path, "patchcore_params.pkl"), "rb") as f:
                self.model_params = pickle.load(f)
            
            infer_shape = self.model_params['input_shape'][1:]
            self.transform_img = transforms.Compose([
                transforms.Resize(infer_shape),
                transforms.CenterCrop(infer_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ])
            
            LOGGER.info("Model Loaded Successfully")
        except Exception as e:
            LOGGER.error(f"Failed to load model: {e}")

    def preprocess(self, img):
        """
        Preprocess the input image.

        Args:
            img (numpy.ndarray): Input image in BGR format.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        img = self.transform_img(img)
        return img

    def infer(self, image, threshold,batch_size = 1):
        """
        Perform inference on a single image.

        Args:
            image (str or numpy.ndarray): Path to the image file or the image array or list of images.
            threshold (float): Threshold for anomaly detection.

        Returns:
            tuple: Score map, masked output, and anomaly score.
        """
        if isinstance(image,list):
            LOGGER.info("Inferening list of images")
            # For list of images
            image_tensors = []
            for img in image:
                image_tensors.append(self.preprocess(img)[None])
            combined_image_tensor = torch.cat(image_tensors)
            batched_tensors = torch.split(combined_image_tensor, batch_size, dim=0)


            segmentations_list = []
            # Batch Inference
            for image_tensor in batched_tensors:
                segmentations_list.append(self.detect(image_tensor, self.patchcore_model))

            segmentations = np.concatenate(segmentations_list,axis=0)
            score_maps = []
            masked_outputs = []
            scores = []
            for segmentation in segmentations:
                score_map, masked_output, score = self.post_process(img, segmentation, threshold)
                score_maps.append(score_map)
                masked_outputs.append(masked_output)
                scores.append(score)
            return score_maps,masked_outputs,scores
            
        else:
            LOGGER.info("Inferening single image")
            # For single image/path
            if os.path.isfile(image):
                img = cv2.imread(image)
            else:
                img = image.copy()
            
            image_tensor = self.preprocess(img)[None]
            segmentations = self.detect(image_tensor, self.patchcore_model)
            score_map, masked_output, score = self.post_process(img, segmentations[0], threshold)
            
            return score_map, masked_output, score

    def predict_score(self, image, PatchCore_list):
        """
        Predict anomaly scores for the given image using multiple PatchCore models.

        Args:
            image (torch.Tensor): Input image tensor.
            PatchCore_list (list): List of PatchCore models.

        Returns:
            numpy.ndarray: Array of segmentations.
        """
        Scores, Segmentations = [], []
        for i, PatchCore in enumerate(PatchCore_list):
            torch.cuda.empty_cache()
            LOGGER.info(f'Predicting score for PatchCore Model {i+1}/{len(PatchCore_list)}...')
            _, masks = PatchCore.predict(image)
            Segmentations.append(masks)

        segmentations = np.array(Segmentations)
        return segmentations

    def post_process(self, img, segmentations, threshold):
        """
        Post-process the segmentation results.

        Args:
            img (numpy.ndarray): Original image.
            segmentations (numpy.ndarray): Segmentation mask.
            threshold (float): Threshold for binarization.

        Returns:
            tuple: Score map, blended output image, and anomaly score.
        """
        img = np.array(img)
        op = segmentations
        op = cv2.resize(op, (img.shape[1], img.shape[0]))
        ret, thresh = cv2.threshold(op, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        masked_img = np.where(thresh, (0, 0, 255), img).astype(np.uint8)
        out = cv2.addWeighted(img, 0.8, masked_img, 0.2, 0)
        anomaly_score = np.max(op)
        LOGGER.info(f'Anomaly score calculated: {anomaly_score}')

        return op, out, anomaly_score

    def detect(self, image, PatchCore_list):
        """
        Detect anomalies in the given image using multiple PatchCore models.

        Args:
            image (torch.Tensor): Input image tensor.
            PatchCore_list (list): List of PatchCore models.

        Returns:
            tuple: Average segmentations.
        """
        LOGGER.info('Starting detection...')
        segmentations = self.predict_score(image, PatchCore_list)
        segmentations = np.mean(segmentations, axis=0)
        return segmentations