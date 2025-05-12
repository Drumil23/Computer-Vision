# Computer Vision Laboratory

This repository contains a collection of computer vision experiments and a mini project focusing on different image processing and computer vision techniques.

## Experiments

1. [Edge Detection](Exp1%20-%20Edge%20Detection/C052_CV_Exp-1.ipynb) - Implementation of various edge detection techniques
2. [Difference of Gaussians (DoG)](Exp2%20-%20Difference%20of%20Gaussians%20(DoG)/C052_CV_Exp-2.ipynb) - Image enhancement using DoG
3. [Histogram of Oriented Gradients (HoG)](Exp3%20-%20Histogram%20of%20Oriented%20Gradients%20(HoG)/) - Feature extraction using HoG
4. [Harris Corner Detector](Exp4%20-%20Harris%20Corner%20Detector/) - Corner detection in images
5. [SIFT](Exp5%20-%20SIFT/) - Scale-Invariant Feature Transform implementation
6. [ORB Descriptor](Exp6%20-%20ORB%20Descriptor/) - Oriented FAST and Rotated BRIEF features
7. [Kmeans](Exp7%20-%20Kmeans/) - Image segmentation using Kmeans clustering
8. [Gabor Filter](Exp8%20-%20Gabor%20Filter/) - Texture analysis using Gabor filters
9. [Gaussian Mixture Model](Exp9%20-%20Gaussian%20Mixture%20Model/) - Probabilistic modeling

## Mini Project: 3D Model Generation from Images using MoGe

The [mini project](Mini%20Project/CV_MOGE.ipynb) demonstrates how to generate 3D models from single images using Microsoft's MoGe (Monocular 3D Generation) model.

### Dependencies

- PyTorch
- OpenCV
- Trimesh
- NumPy
- PIL
- MoGe model

### How it Works

1. The code clones the Microsoft MoGe repository
2. Installs all required dependencies
3. Loads a pre-trained MoGe model from Hugging Face
4. Processes an input image to generate 3D coordinates, depth maps, and normals
5. Creates a textured 3D mesh using the extracted information
6. Exports the 3D model in PLY format

### Sample Usage

```python
# Load the model from huggingface hub
model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()

# Read the input image
input_image = cv2.cvtColor(cv2.imread("input.png"), cv2.COLOR_BGR2RGB)

# Convert image to tensor
input_image_tensor = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

# Infer 3D information
output = model.infer(input_image_tensor, resolution_level=9, apply_mask=True)

# Generate and export mesh
mesh.export('output_model.ply')
```

## Instructions

Each experiment is contained within its own directory with a Jupyter notebook (.ipynb) file that includes detailed instructions, code implementation, and explanations.

To run the experiments:
1. Open the corresponding notebook in Jupyter
2. Execute cells sequentially
3. Follow the instructions within each notebook

For the mini project, a GPU environment is recommended for faster processing.
