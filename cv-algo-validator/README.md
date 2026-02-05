# Segmentation Model Validation Guide

This repository contains the validation algorithm (`algorithm.py`) used to benchmark and verify segmentation models. This guide outlines the requirements for submitting your model as a Docker image and specifies the current input data format required for the validation dataset.

## 1. Model Submission Requirements

To validate your model using this system, your segmentation algorithm must be packaged as a **Docker image**.

* The validation algorithm receives an `image_digest`.
* It will automatically pull your Docker image and execute it against the validation dataset.
* Ensure your container is configured to accept input images and output segmentation masks/coordinates in a standard format (compatible with the evaluation metrics defined in `algorithm.py`).

## 2. Input Data Specifications

Currently, the validation algorithm expects the ground truth segmentation data (input) to be in a specific **JSON format**. This format closely resembles the **VGG Image Annotator (VIA) JSON export** using polygon regions.

### Current Supported Format: VIA-style Polygon JSON

The input file must be a JSON object where:

1. **Keys** are unique identifiers for the image (usually `filename + filesize`).
2. **Values** are objects containing file metadata and a `regions` dictionary.
3. **Regions** contain `shape_attributes` defining the segmentation polygon.
* `name`: Must be `"polygon"`.
* `all_points_x`: An array of integers representing the X coordinates of the polygon vertices.
* `all_points_y`: An array of integers representing the Y coordinates of the polygon vertices.



**Note:** While we plan to support standard formats like COCO or binary masks in the future, **only the format described below is currently valid.**

### JSON Example

Below is a minimal example of the required structure. Your dataset should look like this:

```json
{
  "_MG_2640_16.png1692355": {
    "fileref": "",
    "size": 1692355,
    "filename": "_MG_2640_16.png",
    "base64_img_data": "",
    "file_attributes": {},
    "regions": {
      "0": {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [1027, 1026, 1026, 1025, 1025],
          "all_points_y": [281, 282, 286, 287, 291]
        },
        "region_attributes": {
          "apple_ID": "0134"
        }
      },
      "1": {
        "shape_attributes": {
          "name": "polygon",
          "all_points_x": [900, 901, 905],
          "all_points_y": [315, 316, 320]
        },
        "region_attributes": {
          "apple_ID": "0135"
        }
      }
    }
  },
  "Next_Image_Name.png999999": {
    "filename": "Next_Image_Name.png",
    "regions": {
        "...": "..."
    }
  }
}

```

## 3. Future Support

We are working on extending the validation algorithm (`algorithm.py`) to support additional industry-standard formats, including:

* COCO JSON format.
* Binary/Semantic Segmentation Masks (PNG/JPG).
* YOLO format.

Please ensure your current submissions strictly adhere to the **VIA Polygon** format to ensure successful validation.