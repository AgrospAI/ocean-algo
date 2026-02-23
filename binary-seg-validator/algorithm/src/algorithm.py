import os
import re
import cv2
import json
import zipfile
import subprocess
import numpy as np
from data import InputParameters
from pathlib import Path
from ocean_runner import Algorithm, Config
from logging import getLogger
from typing import Tuple, Any, Optional
from metrics import IoU, dice_coeff, precision_recall_f1, accuracy

algorithm = Algorithm(config=Config(custom_input=InputParameters))
input_parameters: InputParameters = algorithm.job_details.input_parameters
digest = input_parameters.model_image_digest_reference
inference_cmd = input_parameters.inference_cmd

# Dataset data
SOURCE_VOLUME    = '/workspace'
RECOVER_VOLUME   = '/predictions/runs/segment'

logger = getLogger(__name__)

class Algorithm_:
    def __init__(self):
        self._did = str(algorithm.job_details.files._files[0].did)
        self._zipfile = str(algorithm.job_details.files._files[0].input_files[0])
        self.dataset_data: str = ""
        self.gt_masks: list[np.ndarray] = []
        self.pred_masks: list[np.ndarray] = []
        self.metrics: dict[str, list[float]] = {
            'IoU': [],
            'Dice Coefficient': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
            'Accuracy': []
        }
        self.results: Optional[Any] = None


    def load_annotations(self, annotations_file: str):
        with open(f'{SOURCE_VOLUME}/{annotations_file}', 'r') as annotations:
            return json.load(annotations)


    def append_to_mask(self, x_coords, y_coords, mask):
        if len(x_coords) > 0 and len(y_coords) > 0:
            points = np.array(list(zip(x_coords, y_coords)), dtype=np.int32)
            cv2.fillPoly(mask, [points], 1)


    def add_regions_to_mask_scaled(self, annotations, image_size: Tuple[int, int], 
                                match: str, scale_x: float, scale_y: float):
        
        mask = np.zeros(image_size, dtype=np.uint8)
        
        if not match or match not in annotations:
            logger.info(f'No matching annotation found for key: {match}')
            return mask

        apple_regions = max(map(int, list(annotations[match]['regions'].keys())))

        for region in range(apple_regions + 1):
            str_region = str(region)

            if str_region in annotations[match]['regions']:
                x_coords_orig = np.array(annotations[match]['regions'][str_region]['shape_attributes']['all_points_x'])
                y_coords_orig = np.array(annotations[match]['regions'][str_region]['shape_attributes']['all_points_y'])
                
                x_coords = (x_coords_orig * scale_x).astype(np.int32)
                y_coords = (y_coords_orig * scale_y).astype(np.int32)
                
                x_coords = np.clip(x_coords, 0, image_size[1] - 1)
                y_coords = np.clip(y_coords, 0, image_size[0] - 1)
            else:
                x_coords, y_coords = np.array([]), np.array([])

            self.append_to_mask(x_coords, y_coords, mask)   

        return mask
    

    def get_image_ground_truth(self, image_name: str, image_size: Tuple[int, int], 
                            annotations_file: str, original_size: Tuple[int, int] = (1300, 1300)):

        annotations = self.load_annotations(annotations_file)

        pattern = re.compile(rf'^{re.escape(image_name)}\d*$')
        match = next((k for k in annotations.keys() if pattern.match(k)), None)

        if not match:
            raise ValueError(f'No match found for this pattern')

        scale_x = image_size[1] / original_size[1]
        scale_y = image_size[0] / original_size[0]
        
        mask = self.add_regions_to_mask_scaled(annotations, image_size, match, scale_x, scale_y)

        return mask
    

    def run(self) -> "Algorithm":

        with zipfile.ZipFile(self._zipfile, 'r') as zipf:
            zipf.extractall(SOURCE_VOLUME)

        all_files = os.listdir(SOURCE_VOLUME)

        logger.info(f"Found a total of {len(all_files)} elements in the input directory.")

        config_file = next(os.path.join(SOURCE_VOLUME, file) for file in all_files if file.endswith('.txt'))
        self.dataset_data = config_file

        annotations_file = next(file for file in all_files if file.endswith('.json'))
        images = list(filter(lambda file: file not in ['config.txt', annotations_file], all_files))

        try:
            if 'ultralytics' in digest:
                cmd = ['docker', 'exec', 'model', 'yolo', 'predict', 'model=yolov8s-seg.pt', f'source={SOURCE_VOLUME}', 'save=True', f'project={RECOVER_VOLUME}']
                result = subprocess.run(cmd, check=True)
            else:
                cmd = ['docker', 'exec', 'model'] + inference_cmd.split()
                result = subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f'Error launching prediction algorithm. Error code = {e.returncode}')
            exit(1)

        logger.info(f" Prediction path exists in FS? {os.path.exists(f'{RECOVER_VOLUME}/predict')}")
        logger.info(f"{os.listdir(f'{RECOVER_VOLUME}/predict')}")

        masks = [os.path.join(RECOVER_VOLUME, 'predict', mask_name) for mask_name in os.listdir(f'{RECOVER_VOLUME}/predict')]

        logger.info(f"{os.listdir(f'{RECOVER_VOLUME}/predict')}")

        logger.info(f'Found a total of {len(masks)} masks.')
        logger.info(f'Sample of masks: {masks[:5]}.')

        masks_np = []

        for mask_file in masks:
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                _, mask_binary = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                masks_np.append(mask_binary)
            else:
                logger.error(f'Error loading mask: {mask_file}')
        
        logger.info(f'Converted image predictions to binary numpy arrays.')

        masks_gt_np = []

        for image in images:
            current_size = masks_np[0].shape
            img_gt = self.get_image_ground_truth(image, current_size, annotations_file)
            masks_gt_np.append(img_gt)
        
        logger.info(f'Obtained ground truth for test dataset.')

        self.gt_masks = masks_gt_np
        self.pred_masks = masks_np


        for i in range(len(masks_np)):
            iou = IoU(masks_np[i], masks_gt_np[i])
            dice_coefficient = dice_coeff(masks_np[i], masks_gt_np[i])
            precision, recall, f1 = precision_recall_f1(masks_np[i], masks_gt_np[i])
            accuracy_ = accuracy(masks_np[i], masks_gt_np[i])

            if iou == 0 and dice_coefficient == 0 and precision == 0 and recall == 0 and f1 == 0: 
                continue

            self.metrics['IoU'].append(iou)
            self.metrics['Dice Coefficient'].append(dice_coefficient)
            self.metrics['Precision'].append(precision)
            self.metrics['Recall'].append(recall)
            self.metrics['F1'].append(f1)
            self.metrics['Accuracy'].append(accuracy_)
        

        logger.info(f'Generated evaluation with different metrics for test dataset.')

        return self


    def convert(self, o):
        if isinstance(o, np.integer):
            return int(o)
        raise TypeError(f'Object of type {type(o)} is not JSON serializable.')

    def save_result(self) -> "Algorithm":
        with open('/data/outputs/metrics.json', 'w', encoding='utf-8') as metrics_file:
            json.dump(self.metrics, metrics_file, default=self.convert)
        logger.info(f'Metrics file saved at /data/outputs/metrics.json')
        return self


    def generate_executive_summary(self) -> str:
        iou       = np.round(np.mean(self.metrics['IoU']), 2)
        dice      = np.round(np.mean(self.metrics['Dice Coefficient']), 2)
        precision = np.round(np.mean(self.metrics['Precision']), 2)
        recall    = np.round(np.mean(self.metrics['Recall']), 2)
        f1        = np.round(np.mean(self.metrics['F1']), 2)
        accuracy  = np.round(np.mean(self.metrics['Accuracy']), 2)

        # Thresholds for detecting poor balance
        precision_recall_gap = abs(precision - recall)

        if iou >= 0.75:
            if precision_recall_gap > 0.25:
                return f"""
    The algorithm demonstrates high overall segmentation performance, with a strong mean IoU of {iou:.2f}. 
    However, there is a noticeable imbalance between precision ({precision:.2f}) and recall ({recall:.2f}), 
    suggesting the model may overpredict or underpredict certain classes. 
    Despite this, the results are promising and indicate that with minor calibration, the model is ready for deployment.
    """
            else:
                return f"""
    The algorithm exhibits outstanding segmentation performance, with a mean Intersection over Union (IoU) of {iou:.2f}, 
    indicating highly accurate delineation of target regions. Dice coefficient ({dice:.2f}), precision ({precision:.2f}), and recall ({recall:.2f}) 
    are also consistently high, demonstrating a well-balanced and reliable model. 
    This performance exceeds typical benchmarks and supports immediate deployment in production environments.
    """

        elif iou >= 0.5:
            return f"""
    The algorithm delivers solid segmentation results, achieving a mean IoU of {iou:.2f}. 
    Supporting metrics such as Dice coefficient ({dice:.2f}), precision ({precision:.2f}), and recall ({recall:.2f}) indicate balanced and reliable performance. 
    While there is some room for improvement in handling edge cases or rare classes, the model is suitable for deployment with minimal adjustments.
    """

        elif iou >= 0.3:
            return f"""
    The algorithm shows moderate segmentation performance, with a mean IoU of {iou:.2f}. 
    Although recall ({recall:.2f}) may be acceptable, the relatively low IoU and Dice coefficient ({dice:.2f}) suggest inconsistencies in segmenting object boundaries. 
    Further model tuning or data refinement is recommended before deployment.
    """

        else:
            return f"""
    The algorithm currently underperforms in segmentation tasks, with a low mean IoU of {iou:.2f}, indicating poor alignment with ground truth masks. 
    Precision ({precision:.2f}) and recall ({recall:.2f}) metrics suggest fundamental issues in the prediction quality. 
    Substantial improvements in model design, training data, or preprocessing are necessary before considering this model for deployment.
    """


    def build_template(self) -> None:
        metrics_means = {
            '__IOUMEAN__'       : np.round(np.mean(self.metrics['IoU']), 2),
            '__DICEMEAN__'      : np.round(np.mean(self.metrics['Dice Coefficient']), 2),
            '__PRECISIONMEAN__' : np.round(np.mean(self.metrics['Precision']), 2),
            '__RECALLMEAN__'    : np.round(np.mean(self.metrics['Recall']), 2),
            '__F1MEAN__'        : np.round(np.mean(self.metrics['F1']), 2),
            '__ACCURACYMEAN__'  : np.round(np.mean(self.metrics['Accuracy']), 2)
        }

        rand_images_idxs = np.random.randint(low=0, high=len(self.metrics['IoU']), size=5)

        metrics_arrays = {
            '__IOU_ARRAY__'      : f"[{self.metrics['IoU'][rand_images_idxs[0]]}, {self.metrics['IoU'][rand_images_idxs[1]]}, {self.metrics['IoU'][rand_images_idxs[2]]}, {self.metrics['IoU'][rand_images_idxs[3]]}, {self.metrics['IoU'][rand_images_idxs[4]]}]",
            '__DICE_ARRAY__'     : f"[{self.metrics['Dice Coefficient'][rand_images_idxs[0]]}, {self.metrics['Dice Coefficient'][rand_images_idxs[1]]}, {self.metrics['Dice Coefficient'][rand_images_idxs[2]]}, {self.metrics['Dice Coefficient'][rand_images_idxs[3]]}, {self.metrics['Dice Coefficient'][rand_images_idxs[4]]}]",
            '__PRECISION_ARRAY__': f"[{self.metrics['Precision'][rand_images_idxs[0]]}, {self.metrics['Precision'][rand_images_idxs[1]]}, {self.metrics['Precision'][rand_images_idxs[2]]}, {self.metrics['Precision'][rand_images_idxs[3]]}, {self.metrics['Precision'][rand_images_idxs[4]]}]",
            '__RECALL_ARRAY__'   : f"[{self.metrics['Recall'][rand_images_idxs[0]]}, {self.metrics['Recall'][rand_images_idxs[1]]}, {self.metrics['Recall'][rand_images_idxs[2]]}, {self.metrics['Recall'][rand_images_idxs[3]]}, {self.metrics['Recall'][rand_images_idxs[4]]}]",
            '__F1_ARRAY__'       : f"[{self.metrics['F1'][rand_images_idxs[0]]}, {self.metrics['F1'][rand_images_idxs[1]]}, {self.metrics['F1'][rand_images_idxs[2]]}, {self.metrics['F1'][rand_images_idxs[3]]}, {self.metrics['F1'][rand_images_idxs[4]]}]",
            '__ACCURACY_ARRAY__' : f"[{self.metrics['Accuracy'][rand_images_idxs[0]]}, {self.metrics['Accuracy'][rand_images_idxs[1]]}, {self.metrics['Accuracy'][rand_images_idxs[2]]}, {self.metrics['Accuracy'][rand_images_idxs[3]]}, {self.metrics['Accuracy'][rand_images_idxs[4]]}]"
        }


        total_tp = total_tn = total_fp = total_fn = total = 0

        for gt_mask, pred_mask in zip(self.gt_masks, self.pred_masks):
            tp = np.sum((gt_mask == 1) & (pred_mask == 1))
            tn = np.sum((gt_mask == 0) & (pred_mask == 0))
            fp = np.sum((gt_mask == 0) & (pred_mask == 1))
            fn = np.sum((gt_mask == 1) & (pred_mask == 0))
            
            total_tp += tp
            total_tn += tn
            total_fp += fp
            total_fn += fn
            
            total += tp + tn + fp + fn

        total_pixels_label = f'{total / 1e6:.1f}M' if total >= 1e6 else str(total)

        confusion_matrix_data = {
            '__TN_PERCENT__': round(total_tn / total * 100, 1),
            '__FP_PERCENT__': round(total_fp / total * 100, 1),
            '__FN_PERCENT__': round(total_fn / total * 100, 1),
            '__TP_PERCENT__': round(total_tp / total * 100, 1),
            '__TOTAL_PIXELS__': total_pixels_label,
            '__CORRECT_PERCENT__': round((total_tp + total_tn) / total * 100, 1)
        }

        with open(self.dataset_data, 'r', encoding='utf-8') as dd:
            dataset_data = {
                key.strip(): value.strip()
                for line in dd
                for key, value in [line.split('=', 1)]
            }
                
        metadata = {
            '__DATASET_NAME__': dataset_data['DATASET_NAME'],
            '__DATASET_URL__': dataset_data['DATASET_URL']
        }


        # Read HTML Predefined Template
        with open('/algorithm/src/report_template.html', 'r', encoding='utf-8') as report_template:
            html = report_template.read()
        
        executive_summary = self.generate_executive_summary()

        html = html.replace('__EXECUTIVE_SUMMARY__', executive_summary)

        # Replace special markers in the template
        for label, meta in metadata.items():
            html = html.replace(label, str(meta))

        for metric, value in metrics_means.items():
            html = html.replace(metric, str(value))

        for metric, value in metrics_arrays.items():
            html = html.replace(metric, str(value))

        for label, value in confusion_matrix_data.items():
            html = html.replace(label, str(value))

        # Save the new template
        with open(f'/data/outputs/report_template.html', 'w', encoding='utf-8') as report_template:    
            report_template.write(html)

        logger.info(f'Template has been built with the computed metrics.')

algo = Algorithm_().run().save_result()
algo.build_template()


subprocess.run(["docker", "stop", "model"], check=False)
subprocess.run(["docker", "rm", "model"], check=False)
os._exit(0)