# microTVM with PyTorch on STM32 Nucleo L476RG

Welcome to this comprehensive demo where we showcase how to deploy a simple PyTorch model onto an STM32 Nucleo L476RG board using microTVM. This guide will cover everything from the initial model setup in PyTorch to the final deployment on the microcontroller.

## Project Overview

In this project, we train a simple model using CIFAR-10 dataset and integrate it into STM32 IDE environment. For more details about TVM and microTVM, please refer to the [official TVM and microTVM documentation](https://tvm.apache.org/docs/microtvm/index.html).

## Prerequisites

Before you begin, ensure you have the following tools and libraries installed:
- STM32CubeIDE (for firmware flashing)
- Google Colab example provided by TVM:
  (https://colab.research.google.com/github/apache/tvm-site/blob/asf-site/docs/_downloads/1f4943aed1aa607b2775c18b1d71db10/from_pytorch.ipynb)

## Model Definition in PyTorch

Begin by establishing a basic neural network model in PyTorch. Here's an example of a network that stays within the maximum RAM limitations of the STM32 Nucleo L476RG.

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        # First convolutional layer
        self.conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=16, stride=16)
        # Fully connected layer 1
        self.fc = nn.Linear(6 * 2 * 2, 10)

    def forward(self, x):
        # Apply first convolutional layer followed by ReLU and max pooling
        x = self.pool(F.relu(self.conv(x)))
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 6 * 2 * 2)
        # Apply the first fully connected layer followed by ReLU
        x = self.fc(x)
        return x
```

## Model Conversion and Optimization

After defining your model, the next step involves converting the model for deployment and optimizing it using TVM. Here is how to proceed:

### Model Setup and Evaluation

Start by initializing your model on the CPU and loading the pre-trained weights. Then, set the model to evaluation mode.

```python
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
import pathlib

# Create the model on the CPU device
model = SimpleCNN().to(torch.device('cpu'))
model.load_state_dict(torch.load('simpleCNN_v1.pt', map_location=torch.device('cpu')))
model.eval()
```

## Preparing the Input

Prepare a random input based on the input shape expected by the model and trace the model to create a TorchScript object.

```python
import torch

device = torch.device('cpu')
input_shape = [1, 3, 32, 32]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
```

## Data Normalization and Test Loader Setup

Set up data transformations and load the CIFAR-10 test dataset to use with your model.

from torchvision import transforms, datasets
from torch.utils.data import DataLoader

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize each channel
])
testing_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
batch_size = 1  # Adjust as needed
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
```

## Selection of Input image and Conversion to TVM Relay format

```python
input_name = "input"
img = img.numpy()
print(img.shape)
shape_list = [(input_name, input_shape)]
relay_mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
```

## Configuring and Compiling the Model

In this step, we configure the model for optimization and compilation using TVM's advanced features such as CMSIS-NN and USMP (Unified Static Memory Planning). This ensures that the model is optimized for the target device, in this case, a microcontroller with an ARM Cortex-M4 CPU.

Set up the model to use TVM's native schedules or opt for the CMSIS-NN kernels using the Bring-Your-Own-Code (BYOC) capability. This choice can greatly influence performance and efficiency.

```python
# We can use TVM native schedules or rely on the CMSIS-NN kernels using TVM Bring-Your-Own-Code (BYOC) capability.
USE_CMSIS_NN = True

# USMP (Unified Static Memory Planning) performs memory planning of all tensors holistically to achieve best memory utilization
DISABLE_USMP = False

# Use the C runtime (crt)
RUNTIME = Runtime("crt")

# We define the target by passing the board name to `tvm.target.target.micro`.
# If your board is not included in the supported models, you can define the target such as:
TARGET = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m4")
# TARGET = tvm.target.target.micro("stm32l4r5zi")

# Use the AOT executor rather than graph or vm executors. Use unpacked API and C calling style.
EXECUTOR = tvm.relay.backend.Executor(
    "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8}
)

# Now, we set the compilation configurations and compile the model for the target:
config = {"tir.disable_vectorize": True}
if USE_CMSIS_NN:
    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
if DISABLE_USMP:
    config["tir.usmp.enable"] = False

with tvm.transform.PassContext(opt_level=3, config=config):
    if USE_CMSIS_NN:
        # When we are using CMSIS-NN, TVM searches for patterns in the
        # relay graph that it can offload to the CMSIS-NN kernels.
        relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)
    lowered = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )
parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
print(f"Model parameter size: {parameter_size}")

# We need to pick a directory where our file will be saved.
# If running on Google Colab, we'll save everything in ``/root/tutorial`` (aka ``~/tutorial``)
# but you'll probably want to store it elsewhere if running locally.

BUILD_DIR = pathlib.Path("/content/")

BUILD_DIR.mkdir(exist_ok=True)

# Now, we export the model into a tar file:
TAR_PATH = pathlib.Path(BUILD_DIR) / "model_pytorch.tar"
export_model_library_format(lowered, TAR_PATH)
```

## Integrate sample header file into the TAR file

This code snippet prepares a directory to store sample images, loads the CIFAR-10 dataset with a specified transformation that includes tensor conversion and normalization, and iteratively tests the model to identify correctly classified images. For each correctly classified image, it extracts the image, prints its label, and archives it in a TAR file with an appropriate filename based on its classification.

```python
SAMPLES_DIR = "samples"

# Ensure the samples directory exists
if not os.path.exists(SAMPLES_DIR):
    os.makedirs(SAMPLES_DIR)

# CIFAR-10 classes
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Transformation: Convert to tensor, then apply a simple normalization by multiplying by 0.5
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize each channel
])

dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

for images, labels in test_loader:
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      if(predicted == labels):
        img = images
        print(f"correct label is{labels}")
        if(labels == 0):
          break

with tarfile.open(TAR_PATH, mode="a") as tar_file:
  SAMPLES_DIR = "samples"
  print(np.asarray(img))
  # Create file name including the class name and index
  file_name = "sample_0"
  create_header_file(file_name, np.asarray(img), SAMPLES_DIR, tar_file)
```
## Integrate the model.tar file into your custom IDE provided by microTVM

Please follow the instructions provided by TVM for this step:
(https://tvm.apache.org/docs/how_to/work_with_microtvm/micro_custom_ide.html)

## STM32 Cube IDE 

UART2 is enabled by default which can be used to print Inference Time and Classification Results

<img width="652" alt="Screenshot 2024-04-13 at 7 56 50 PM" src="https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/e9eb72b5-7f60-47d9-9384-a58575a90135">

## Live Demo in Nucleo L476RG board [ Airplane, Automobile ]

https://github.com/adithya2424/EmbeddedEdgeAI/assets/34277400/0aa1729b-e59d-43d2-848f-931b2dcf5c34

## Final Remarks

The testcnn.zip file includes the entire project directory, set up for use in the STM32 Cube IDE environment. To incorporate additional sample data into the current project structure, utilize the Linker script to allocate the new sample data to RAM2.


Crafted with love ❤️ and passion for the Embedded Edge AI. Let's innovate together! 🚀✨


















