# main.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from model import BasicCNN
from pruning import fine_grained_prune
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
from pruning import FineGrainedPruner
from model_metrics import *
import copy
from quantize import *

def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(2, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(40, 80)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend([
                'accuracy after pruning',
                f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
            ])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) \
                          in model.named_parameters() if param.dim() > 1]
    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(sparsities, desc=f'scanning {i_layer}/{len(named_conv_weights)} weight - {name}'):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = test(None, model, dataloader, device)
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')
            # restore
            param.copy_(param_clone)
            accuracy.append(acc)
        if verbose:
            print(
                f'\r    sparsity=[{",".join(["{:.2f}".format(x) for x in sparsities])}]: accuracy=[{", ".join(["{:.2f}%".format(x) for x in accuracy])}]',
                end='')
        accuracies.append(accuracy)
    return sparsities, accuracies


def train(args, model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{args.epochs}], Loss: {running_loss / len(train_loader)}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(f'Accuracy of the model on the validation images: {100 * correct / total} %')

    # Save the trained model
    torch.save(model.state_dict(), 'BasicCNN_model.pth')

def train_prune(model, trainloader, criterion, optimizer, callbacks=None):
    model.train()
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if callbacks is not None:
            for callback in callbacks:
                callback()
    print('Finished Training')

def test(args, model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total} %')
    return 100 * correct / total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or Test a Basic CNN on CIFAR-10')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'sensitivity', 'param_distribution', 'print_weights', 'FineGrainedPruner', 'Finetune', 'Compare_models', 'quantize'], help='Mode: train or test')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training/testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    args.mode = 'quantize'
    if args.mode == 'train':
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        model = BasicCNN(n_classes=10).to(device)
        train(args, model, train_loader, val_loader, device)
    elif args.mode == 'test':
        model = BasicCNN(10).to(device)
        model.eval()
        model.load_state_dict(torch.load('BasicCNN_Pruned_Finetuned_model.pth'))
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        test(args, model, test_loader, device)
    elif args.mode == 'sensitivity':
        model = BasicCNN(10).to(device)
        model.eval()
        model.load_state_dict(torch.load('BasicCNN_model.pth'))
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        sparsities, accuracies = sensitivity_scan(
            model, test_loader, scan_step=0.1, scan_start=0.4, scan_end=1.0)
        model_accuracy = test(args, model, test_loader, device)
        plot_sensitivity_scan(sparsities, accuracies, model_accuracy)
    elif args.mode == 'param_distribution':
        model = BasicCNN(10).to(device)
        model.eval()
        model.load_state_dict(torch.load('BasicCNN_model.pth'))
        plot_num_parameters_distribution(model)
        plot_weight_distribution(model)
    elif args.mode == 'FineGrainedPruner':
        model = BasicCNN(10).to(device)
        model.eval()
        model.load_state_dict(torch.load('BasicCNN_model.pth'))
        dense_model_size = get_model_size(model)
        sparsity_dict = {
            'conv1.weight': 0.4,
            'conv2.weight': 0.6,
            'fc1.weight': 0.67,
            'fc2.weight': 0.5
        }
        pruner = FineGrainedPruner(model, sparsity_dict)
        print(f'After pruning with sparsity dictionary')
        for name, sparsity in sparsity_dict.items():
            print(f'  {name}: {sparsity:.2f}')
        print(f'The sparsity of each layer becomes')
        for name, param in model.named_parameters():
            if name in sparsity_dict:
                print(f'  {name}: {get_sparsity(param):.2f}')
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        sparse_model_size = get_model_size(model, count_nonzero_only=True)
        print(
            f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
        sparse_model_accuracy = test(args, model, test_loader, device)
        print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% before fintuning")
        plot_weight_distribution(model, count_nonzero_only=True)
        #  save pruned model
        torch.save(model.state_dict(), 'BasicCNN_Pruned_model.pth')
    elif args.mode == 'print_weights':
        model = BasicCNN(10)
        model.load_state_dict(torch.load('BasicCNN_model.pth'))
        # Print the names and shapes of weights using state_dict
        for name, param in model.state_dict().items():
            print(f"Layer: {name}, Shape: {param.size()}")
    elif args.mode == 'Finetune':
        # model old is just used to compute dense model size i.e. original model size
        model_old = BasicCNN(10).to(device)
        model_old.eval()
        model_old.load_state_dict(torch.load('BasicCNN_model.pth'))
        dense_model_size = get_model_size(model_old)
        model = BasicCNN(10).to(device)
        model.load_state_dict(torch.load('BasicCNN_Pruned_model.pth'))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        num_finetune_epochs = 5
        best_sparse_model_checkpoint = dict()
        best_accuracy = 0
        print(f'Finetuning Fine-grained Pruned Sparse Model')
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        sparsity_dict = {
            'conv1.weight': 0.4,
            'conv2.weight': 0.6,
            'fc1.weight': 0.67,
            'fc2.weight': 0.5
        }
        pruner = FineGrainedPruner(model, sparsity_dict)
        for epoch in range(num_finetune_epochs):
            # At the end of each train iteration, we have to apply the pruning mask
            #    to keep the model sparse during the training
            train_prune(model, train_loader, criterion, optimizer,
                        callbacks=[lambda: pruner.apply(model)])
            accuracy = test(args, model, test_loader, device)
            is_best = accuracy > best_accuracy
            if is_best:
                best_sparse_model_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
                best_accuracy = accuracy
            print(f'    Epoch {epoch + 1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')

        model.load_state_dict(best_sparse_model_checkpoint['state_dict'])
        sparse_model_size = get_model_size(model, count_nonzero_only=True)
        print(
            f"Sparse model has size={sparse_model_size / MiB:.2f} MiB = {sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
        sparse_model_accuracy = test(args, model, test_loader, device)
        print(f"Sparse model has accuracy={sparse_model_accuracy:.2f}% after fintuning")
        torch.save(model.state_dict(), 'BasicCNN_Pruned_Finetuned_model.pth')
    elif args.mode == 'Compare_models':
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        model = BasicCNN(10).to(device)
        model.eval()
        model.load_state_dict(torch.load('BasicCNN_model.pth'))
        dense_model_size = get_model_size(model)
        print(f"Baseline model size={dense_model_size / MiB:.2f} MiB")
        test(args, model, test_loader, device)
        model.load_state_dict(torch.load('BasicCNN_Pruned_model.pth'))
        pruned_model_size = get_model_size(model, count_nonzero_only=True)
        print(f"Pruned model size={pruned_model_size / MiB:.2f} MiB")
        test(args, model, test_loader, device)
        model.load_state_dict(torch.load('BasicCNN_Pruned_Finetuned_model.pth'))
        Finetuned_Pruned_model_size = get_model_size(model, count_nonzero_only=True)
        print(f"Finetuned + Pruned model size={Finetuned_Pruned_model_size / MiB:.2f} MiB")
        test(args, model, test_loader, device)
    elif args.mode == "quantize":
        print('Note that the storage for codebooks is ignored when calculating the model size.')
        quantizers = dict()
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        for bitwidth in [8, 4, 2]:
            model = BasicCNN(10).to(device)
            model.eval()
            model.load_state_dict(torch.load('models/BasicCNN_Pruned_Finetuned_model.pth'))
            print(f'k-means quantizing model into {bitwidth} bits')
            quantizer = KMeansQuantizer(model, bitwidth)
            quantized_model_size = get_model_size(model, bitwidth)
            print(f"    {bitwidth}-bit k-means quantized model has size={quantized_model_size / MiB:.2f} MiB")
            quantized_model_accuracy = test(args, model, test_loader, device)
            print(f"    {bitwidth}-bit k-means quantized model has accuracy={quantized_model_accuracy:.2f}%")
            quantizers[bitwidth] = quantizer









