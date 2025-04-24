### **Model Pruning Experiment:**

This experiment demonstrates our latest efforts in fine-tuning and pruning a neural network model to achieve optimal performance on resource-constrained devices.

## **Experiment Setup**

The main components of our experiment include:

- **BasicCNN:** A simple yet powerful CNN architecture designed for image classification tasks.
- **FineGrainedPruner:** A tool for applying fine-grained pruning to selectively remove weights from the model based on certain criteria, aiming to reduce model size while maintaining accuracy.
- **Sensitivity Analysis:** An analysis technique to understand the impact of various sparsity levels on model performance.

## **Getting Started**

To replicate our experiment or to use the pruning techniques in your own projects, follow these steps:

### Prerequisites

Ensure you have the following installed:
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/adithya2424/EmbeddedEdgeAI.git
cd EmbeddedEdgeAI
```

# Fine-Grained Pruning Guide

This guide outlines a structured approach for implementing fine-grained pruning on neural network models, aimed at reducing model size and computational complexity while maintaining high performance. The process involves analyzing weight distribution, performing sensitivity analysis, examining parameter distribution, creating a sparsity dictionary, applying pruning, and fine-tuning the pruned model.

## Step 1: Analyze Model Weight Distribution

Understanding your model's weight distribution is crucial before beginning the pruning process. This analysis helps identify layers that may be more amenable to pruning.

- **Action:** Visualize the weight distribution across the model using the `plot_weight_distribution` function.
 ![Weightdistribution](https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/9fd3769b-5db7-4aed-b3df-9479206ba9c0)
 
## Step 2: Perform Sensitivity Analysis

Sensitivity analysis assesses the impact of varying levels of sparsity on different layers' performance, helping to identify how resilient each layer is to weight removal.

- **Action:** Use the `sensitivity_scan` function to evaluate how different sparsity levels affect model accuracy.
  ![sensitivity](https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/c60119af-da7b-4ac1-a0c0-d7a1a75a5154)

## Step 3: Examine Model Parameters Distribution

A closer look at the distribution of parameters can reveal layers with excess redundancy or minimal contribution to the model's output, indicating potential pruning candidates.

- **Action:** Analyze the distribution of parameters using `plot_num_parameters_distribution` and `plot_weight_distribution`.
![Paramdistribution](https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/659dad35-02cb-4770-87dc-6fff711a5e9d)

## Step 4: Create a Manual Sparsity Dictionary

Based on the insights gained, create a sparsity dictionary specifying desired sparsity levels for each layer, allowing for a tailored pruning approach.

- **Example:**
  ```python
  sparsity_dict = {
      'conv1.weight': 0.4,  # 40% sparsity
      'conv2.weight': 0.6,  # 60% sparsity
      'fc1.weight': 0.5,    # 50% sparsity
      'fc2.weight': 0.7     # 70% sparsity
  }

## Model Comparison Table

The following table compares the old (baseline) model, the pruned model, and the fine-tuned plus pruned model in terms of memory usage and accuracy. Note that the reported memory usage for the pruned and fine-tuned + pruned models reflects the effective memory usage by counting only non-zero parameters, which illustrates the theoretical memory savings. The actual file size on disk may not decrease correspondingly without using a storage format that supports sparse representations.

| Model Type                 | Memory Usage (MB) | Accuracy (%) |
|----------------------------|-------------------|--------------|
| Old (Baseline) Model       | 1.00              | 69.46        |
| Pruned Model               | 0.33              | 65.88        |
| Fine-Tuned + Pruned Model  | 0.33              | 70.10        |

*Memory usage is calculated based on counting only the non-zero parameters to demonstrate the potential reduction in model complexity. Actual disk storage savings require a format that efficiently represents sparse matrices.*

## Acknowledgments

This project incorporates code from the course "MIT 6.5940 EfficientML.ai," offered in Fall 2023. The course provided valuable insights and examples that have been adapted and used in this project. I extend my gratitude to the course instructors and MIT for making such resources available.

For more information about the course and its materials, please visit [MIT 6.5940 EfficientML.ai Fall 2023](https://hanlab.mit.edu/courses/2023-fall-65940) or the course's direct link if available.














