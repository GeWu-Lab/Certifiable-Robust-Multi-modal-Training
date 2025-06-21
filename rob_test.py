import torch
from torch import nn
import sklearn.metrics
import numpy as np
from torchvision import transforms

eps = 1e-13

def accuracy(truth, pred):
    """Computes accuracy score."""
    return sklearn.metrics.accuracy_score(truth.cpu().numpy(), pred.cpu().numpy())

def get_pred(out):
    """Gets the prediction from the model output."""
    return torch.argmax(out, 1).detach()

def evaluate(pred, true):
    """
    Evaluates the predictions against the ground truth.
    Args:
        pred (list[Tensor]): A list of prediction tensors from each batch.
        true (list[Tensor]): A list of ground truth tensors from each batch.
    Returns:
        float: The accuracy score.
    """
    if not pred:
        return 0.0
    pred = torch.cat(pred, 0)
    true = torch.cat(true, 0)
    return accuracy(true, pred)


def obtain_input(batch):
    """Extracts tensors from the batch and moves them to the GPU."""
    visual = batch['clip'].cuda()
    audio = batch['audio'].cuda()
    targets = batch['target'].cuda()
    return visual, audio, targets

def norm_tensor(x, mean, std):
    """
    Normalizes a tensor. Assumes tensor is of shape [B, C, T, H, W].
    Normalization is applied to each frame (T dimension) independently.
    """
    norm = transforms.Normalize(
        mean=mean,
        std=std
    )
    # x is of shape [B, C, T, H, W]
    # The normalization is applied to each frame, which is a [C, H, W] tensor.
    for i in range(x.size(0)):
        for j in range(x.size(2)) :
            x[i, :, j, :, :] = norm(x[i, :, j, :, :])
    return x

class UnNormalize(object):
    """Un-normalizes a tensor given mean and standard deviation."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (C, H, W) to be un-normalized.
        Returns:
            Tensor: Un-normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The original normalize operation is: t.sub_(m).div_(s)
        return tensor

def inv_norm_tensor(x, mean, std):
    """
    Un-normalizes a tensor. Assumes tensor is of shape [B, C, T, H, W].
    Un-normalization is applied to each frame (T dimension) independently.
    """
    inv_norm = UnNormalize(
        mean=mean,
        std=std
    )
    # x is of shape [B, C, T, H, W]
    # The inverse normalization is applied to each frame.
    for i in range(x.size(0)):
        for j in range(x.size(2)):
            x[i, :, j, :, :] = inv_norm(x[i, :, j, :, :])

    return x

def clamp_input(i_adv, modality_type="visual"):
    """
    Clamps the adversarial input to a valid range [0, 1] in pixel space,
    then transforms it back to the normalized space.
    """
    if modality_type == "visual" :
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif modality_type == "flow" :
        mean = [0.1307, 0.1307]
        std = [0.3081, 0.3081]
    else:
        # For modalities like audio, no clamping is applied.
        return i_adv
    
    # Invert normalization to get pixel values
    frame_adv = inv_norm_tensor(i_adv.detach().clone(), mean=mean, std=std)
    # Clamp to the valid [0, 1] range
    frame_adv_clamp = torch.clamp(frame_adv, 0, 1)
    # Re-apply normalization
    i_adv_clamp = norm_tensor(frame_adv_clamp, mean=mean, std=std)
    return i_adv_clamp

def l2_grad(model, data, labels):
    """Computes the L2 gradient of the loss with respect to each input modality."""
    criterion = nn.CrossEntropyLoss()
    modal_num = len(data)
    _, _, preds = model(data[0], data[1])                
    model.zero_grad()
    loss = criterion(preds, labels)
    loss.backward()
    grad = []
    for v in range(modal_num):
        grad.append(data[v].grad.data.reshape(labels.shape[0], -1)) 
    return grad

def l2_attack_one_step(model, data, labels, epsilon, modal_property):
    """
    Performs a single-step L2 attack (like FGSM but with L2 normalization).
    The perturbation is normalized across all modalities combined.
    """
    modal_num = len(data)
    data_adv = []
    grad = l2_grad(model, data, labels)
    grad_norm = torch.zeros((labels.shape[0], 1), device=grad[0].device)
    # Calculate the L2 norm of the concatenated gradients from all modalities
    for modality in range(modal_num):
        grad_norm = grad_norm + (torch.norm(grad[modality], dim=1) ** 2).reshape(grad_norm.shape)
    grad_norm = torch.sqrt(grad_norm) + eps

    for modality in range(modal_num):
        # Normalize gradient and apply perturbation
        grad[modality] = grad[modality] / (grad_norm.expand(grad[modality].shape))
        grad[modality] = grad[modality].view(data[modality].shape)
        i_adv = data[modality].detach() + epsilon * grad[modality].detach()
        data_adv.append(i_adv)
    return data_adv

def modality_specific_l2_attack_one_step(model, data, labels, epsilons, modal_property):
    """
    Performs a single-step L2 attack where the perturbation for each modality
    is normalized independently.
    """
    modal_num = len(data)
    data_adv = []
    grad = l2_grad(model, data, labels)
    for modality in range(modal_num):
        # Calculate L2 norm for the current modality's gradient
        grad_norm = torch.norm(grad[modality], dim=1).reshape(labels.shape[0], 1)
        grad[modality] = grad[modality] / (grad_norm.expand(grad[modality].shape))
        grad[modality] = grad[modality].view(data[modality].shape)
        # Apply modality-specific perturbation
        i_adv = data[modality].detach() + epsilons[modality] * grad[modality].detach()
        data_adv.append(i_adv)
    return data_adv


def multimodal_attack(model, test_dataloader, 
                    attack_type="fgm", modal_property=["visual", "audio"], factor=0.2):
    """
    Performs a multimodal adversarial attack.
    Args:
        model: The model to attack.
        test_dataloader: Dataloader for the test set.
        attack_type (str): "fgm" (single-step) or "pgd_l2" (multi-step).
        modal_property (list): List of modality types.
        factor (float): The epsilon value for the attack.
    """
    model.eval()
    epsilon = factor
    adv_pred_all = []
    labels_all = []

    for i, batch in enumerate(test_dataloader):
        visual, audio, labels = obtain_input(batch)
        data = [visual, audio]
        modal_num = len(data)
        labels_all.append(labels)

        if attack_type == "fgm":
            for v in range(modal_num):
                data[v].requires_grad = True
            data_adv = l2_attack_one_step(model, data, labels, epsilon, modal_property)
            for v in range(modal_num):
                data_adv[v] = clamp_input(data_adv[v], modal_property[v])
        
        elif attack_type == "pgd_l2":
            alpha = epsilon / 3  # Step size for each iteration
            data_adv = [d.clone().detach() for d in data] # Start with clean data
            
            # PGD iterations
            for t in range(4):
                for v in range(modal_num):
                    data_adv[v].requires_grad = True
                # Perform one gradient step
                data_adv = l2_attack_one_step(model, data_adv, labels, alpha, modal_property)
                # Project the perturbation back to the L2 ball
                data_adv = restrict_adv(data_adv, data, epsilon, modal_property)
        else:
            print("Unknown attack method!")
            
        _, _, adv_preds = model(data_adv[0], data_adv[1])
        adv_pred_all.append(get_pred(adv_preds))
    return evaluate(adv_pred_all, labels_all)


def restrict_adv(data_adv, data, epsilon, modal_property):
    """
    Projects the adversarial perturbation back onto an L2 ball of radius epsilon.
    This is the core projection step for PGD.
    """
    modal_num = len(data_adv)
    batch_size = data_adv[0].shape[0]

    # Calculate the perturbation for each modality
    perturbations = [adv - orig for adv, orig in zip(data_adv, data)]

    # Calculate the squared L2 norm of the total perturbation across all modalities
    total_pert_norm_sq = torch.zeros(batch_size, device=data_adv[0].device)
    for v in range(modal_num):
        pert_flat = perturbations[v].view(batch_size, -1)
        total_pert_norm_sq += torch.sum(pert_flat ** 2, dim=1)
    
    total_pert_norm = torch.sqrt(total_pert_norm_sq)

    # Calculate the projection factor.
    # If the perturbation is within the budget, factor is >= 1. We cap it at 1.
    # If it's outside, factor is < 1, scaling it down.
    factor = epsilon / (total_pert_norm + eps)
    factor = torch.min(factor, torch.ones_like(factor))

    new_data_adv = []
    for v in range(modal_num):
        # Reshape factor to allow broadcasting
        factor_reshaped = factor.view(batch_size, *([1] * (perturbations[v].dim() - 1)))
        
        # Project the perturbation
        projected_pert = perturbations[v] * factor_reshaped
        i_adv = data[v] + projected_pert
        
        # Clamp the projected adversarial example to a valid data range
        i_adv_clamped = clamp_input(i_adv, modal_property[v])
        new_data_adv.append(i_adv_clamped)

    return new_data_adv


def unimodal_attack(model, test_dataloader, attack_type="fgm", modal_property=["visual", "audio"], factor=[0, 0.1]):
    """
    Performs a unimodal or modality-specific adversarial attack.
    Args:
        model: The model to attack.
        test_dataloader: Dataloader for the test set.
        attack_type (str): "fgm" or "pgd_l2".
        modal_property (list): List of modality types.
        factor (list[float]): List of epsilon values for each modality.
    """
    model.eval()
    epsilons = np.array(factor)
    adv_pred_all = []
    labels_all = []
    # The total perturbation budget is the L2 norm of individual budgets
    epsilon = np.sqrt(np.sum(epsilons**2))
    for iter, batch in enumerate(test_dataloader):
        visual, audio, labels = obtain_input(batch)
        data = [visual, audio]
        modal_num = len(data) 
        labels_all.append(labels)
        if attack_type == "fgm":
            for v in range(modal_num):
                data[v].requires_grad = True
            data_adv = modality_specific_l2_attack_one_step(model, data, labels, epsilons, modal_property)
            for v in range(modal_num):
                data_adv[v] = clamp_input(data_adv[v], modal_property[v])
        elif attack_type == "pgd_l2":
            alpha = epsilons / 3
            data_adv = [d.clone().detach() for d in data]
            
            for t in range(4):
                for v in range(modal_num):
                    data_adv[v].requires_grad = True 
                # Use modality-specific step
                data_adv = modality_specific_l2_attack_one_step(model, data_adv, labels, alpha, modal_property)
                # Project based on the combined epsilon budget
                data_adv = restrict_adv(data_adv, data, epsilon, modal_property)
        else:
            print("Unknown attack method!")
        _, _, adv_preds = model(data_adv[0], data_adv[1])
        adv_pred_all.append(get_pred(adv_preds))
    return evaluate(adv_pred_all, labels_all)
