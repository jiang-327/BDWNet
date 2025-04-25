import os
import torch
import cv2
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
try:
    from models.network import WoodDefectBD
    from utils.dataset import WoodDefectDataset
    from utils.metrics import calculate_iou, calculate_dice, calculate_precision_recall, calculate_boundary_accuracy
    from configs.config import Config
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False
def check_dependencies():
    if not DEPENDENCIES_AVAILABLE:
        return False
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")
    return True
def calculate_pixel_accuracy(pred, target):
    correct = (pred == target).float().sum()
    total = torch.ones_like(target).sum()
    return correct / total
def calculate_class_pixel_accuracy(pred, target):
    if target.sum() > 0:
        fg_correct = ((pred == 1) & (target == 1)).float().sum()
        fg_total = (target == 1).float().sum()
        fg_acc = (fg_correct / fg_total).item()
    else:
        fg_acc = 1.0 if pred.sum() == 0 else 0.0
    bg_correct = ((pred == 0) & (target == 0)).float().sum()
    bg_total = (target == 0).float().sum()
    bg_acc = (bg_correct / bg_total).item()
    return [fg_acc, bg_acc]
def calculate_mean_iou(pred, target, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum() - intersection
        if union > 0:
            iou = (intersection / union).item()
        else:
            iou = 1.0
        ious.append(iou)
    return np.mean(ious)
def visualize_results(images, masks, predictions, prob_maps, edge_maps=None, save_path=None):
    try:
        num_samples = min(3, len(images))
        num_cols = 5 if edge_maps is None else 6
        if num_samples == 1:
            fig, axes = plt.subplots(1, num_cols, figsize=(20, 5))
        else:
            fig, axes = plt.subplots(num_samples, num_cols, figsize=(20, 5*num_samples))
        for i in range(num_samples):
            if num_samples == 1:
                current_axes = axes
            else:
                current_axes = axes[i]
            img = images[i].cpu()
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            elif img.shape[0] > 3:
                img = img[:3]
            img = img.permute(1, 2, 0).numpy()
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min())
            current_axes[0].imshow(img)
            current_axes[0].set_title('Original Image')
            current_axes[0].axis('off')
            mask = masks[i].cpu().numpy()
            current_axes[1].imshow(mask, cmap='gray')
            current_axes[1].set_title('Ground Truth Mask')
            current_axes[1].axis('off')
            pred = predictions[i].cpu().numpy()
            current_axes[2].imshow(pred, cmap='gray')
            current_axes[2].set_title('Predicted Mask')
            current_axes[2].axis('off')
            prob = prob_maps[i].cpu()
            if len(prob.shape) == 3 and prob.shape[0] == 1:
                prob = prob.squeeze(0)
            prob = prob.numpy()
            current_axes[3].imshow(prob, cmap='jet')
            current_axes[3].set_title('Probability Map')
            current_axes[3].axis('off')
            edge_col_idx = 4
            if edge_maps is not None:
                edge = edge_maps[i].cpu()
                if len(edge.shape) == 3 and edge.shape[0] == 1:
                    edge = edge.squeeze(0)
                edge = edge.numpy()
                current_axes[4].imshow(edge, cmap='jet')
                current_axes[4].set_title('Edge Map')
                current_axes[4].axis('off')
                edge_col_idx = 5
            contour_img = img.copy()
            pred_uint8 = (pred * 255).astype(np.uint8)
            contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_img, contours, -1, (1, 0, 0), 2)
            current_axes[edge_col_idx].imshow(contour_img)
            current_axes[edge_col_idx].set_title('Segmentation Contour')
            current_axes[edge_col_idx].axis('off')
        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"Visualization results saved to: {save_path}")
            except Exception as e:
                print(f"Failed to save visualization results: {e}")
            for i in range(num_samples):
                try:
                    sample_dir = os.path.dirname(save_path)
                    sample_dir = os.path.join(sample_dir, f'sample{i+1}')
                    os.makedirs(sample_dir, exist_ok=True)
                    img = images[i].cpu()
                    if img.shape[0] == 1:
                        img = img.repeat(3, 1, 1)
                    elif img.shape[0] > 3:
                        img = img[:3]
                    img = img.permute(1, 2, 0).numpy()
                    if img.max() > 0:
                        img = (img - img.min()) / (img.max() - img.min())
                    img_uint8 = (img * 255).astype(np.uint8)
                    orig_path = os.path.join(sample_dir, "original.png")
                    try:
                        cv2.imwrite(orig_path, cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))
                        print(f"Original image saved to: {orig_path}")
                    except Exception as e:
                        print(f"Failed to save original image: {e}")
                    mask = masks[i].cpu().numpy()
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_path = os.path.join(sample_dir, "mask.png")
                    try:
                        cv2.imwrite(mask_path, mask_uint8)
                        print(f"Mask saved to: {mask_path}")
                    except Exception as e:
                        print(f"Failed to save mask: {e}")
                    pred = predictions[i].cpu().numpy()
                    pred_uint8 = (pred * 255).astype(np.uint8)
                    pred_path = os.path.join(sample_dir, "prediction.png")
                    try:
                        cv2.imwrite(pred_path, pred_uint8)
                        print(f"Predicted mask saved to: {pred_path}")
                    except Exception as e:
                        print(f"Failed to save predicted mask: {e}")
                    prob = prob_maps[i].cpu()
                    if len(prob.shape) == 3 and prob.shape[0] == 1:
                        prob = prob.squeeze(0)
                    prob = prob.numpy()
                    prob_norm = (prob * 255).astype(np.uint8)
                    prob_color = cv2.applyColorMap(prob_norm, cv2.COLORMAP_JET)
                    prob_path = os.path.join(sample_dir, "probability_map.png")
                    try:
                        cv2.imwrite(prob_path, prob_color)
                        print(f"Probability map saved to: {prob_path}")
                    except Exception as e:
                        print(f"Failed to save probability map: {e}")
                    if edge_maps is not None:
                        edge = edge_maps[i].cpu()
                        if len(edge.shape) == 3 and edge.shape[0] == 1:
                            edge = edge.squeeze(0)
                        edge = edge.numpy()
                        edge_norm = (edge * 255).astype(np.uint8)
                        edge_color = cv2.applyColorMap(edge_norm, cv2.COLORMAP_JET)
                        edge_path = os.path.join(sample_dir, "boundary_map.png")
                        try:
                            cv2.imwrite(edge_path, edge_color)
                            print(f"Boundary map saved to: {edge_path}")
                        except Exception as e:
                            print(f"Failed to save boundary map: {e}")
                    contour_img = img.copy()
                    contours, _ = cv2.findContours(pred_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contour_img_uint8 = (contour_img * 255).astype(np.uint8)
                    cv2.drawContours(contour_img_uint8, contours, -1, (255, 0, 0), 2)
                    contour_path = os.path.join(sample_dir, "contour_result.png")
                    try:
                        cv2.imwrite(contour_path, cv2.cvtColor(contour_img_uint8, cv2.COLOR_RGB2BGR))
                        print(f"Contour image saved to: {contour_path}")
                    except Exception as e:
                        print(f"Failed to save contour image: {e}")
                    print(f"Sample {i+1} images saved to: {sample_dir}")
                except Exception as e:
                    print(f"Error processing sample {i+1}: {e}")
        plt.close()
    except Exception as e:
        print(f"Error during visualization: {e}")
        plt.close()
def test(model, loader, device, save_dir=None, visualize=True, max_vis_samples=3):
    model.eval()
    ious = []
    mious = []
    dices = []
    pas = []
    mpas = []
    aps = []
    boundary_accs = []
    total_time = 0
    total_frames = 0
    fps_list = []
    vis_images = []
    vis_masks = []
    vis_preds = []
    vis_probs = []
    vis_edges = []
    vis_thresh_maps = []
    vis_count = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)
            batch_size = images.size(0)
            total_frames += batch_size
            start_time = time.time()
            try:
                outputs = model(images)
                end_time = time.time()
                inference_time = end_time - start_time
                total_time += inference_time
                if inference_time > 0:
                    batch_fps = batch_size / inference_time
                    fps_list.append(batch_fps)
                    if i % 10 == 0:
                        print(f"Batch {i+1} - FPS: {batch_fps:.2f}")
                pred_masks = outputs.get('binary_map', None)
                if pred_masks is None:
                    print(f"Warning: No binary_map in model output, available keys: {outputs.keys()}")
                    continue
                if len(pred_masks.shape) == 4 and pred_masks.shape[1] == 1:
                    pred_masks = pred_masks.squeeze(1)
                if len(masks.shape) == 4 and masks.shape[1] == 1:
                    masks = masks.squeeze(1)
                thresholded_masks = (pred_masks > 0.5).float()
                batch_pa = calculate_pixel_accuracy(thresholded_masks, masks).detach().cpu().item()
                pas.append(batch_pa)
                for j in range(pred_masks.size(0)):
                    class_accs = calculate_class_pixel_accuracy(thresholded_masks[j], masks[j])
                    mpas.append(np.mean(class_accs))
                batch_iou = calculate_iou(thresholded_masks, masks).detach().cpu().item()
                ious.append(batch_iou)
                for j in range(pred_masks.size(0)):
                    batch_miou = calculate_mean_iou(thresholded_masks[j], masks[j])
                    mious.append(batch_miou)
                batch_dice = calculate_dice(thresholded_masks, masks).detach().cpu().item()
                dices.append(batch_dice)
                for j in range(pred_masks.size(0)):
                    has_positive = masks[j].sum() > 0
                    if has_positive:
                        precision, recall, ap = calculate_precision_recall(pred_masks[j], masks[j])
                        aps.append(float(ap))
                    else:
                        if thresholded_masks[j].sum() == 0:
                            aps.append(1.0)
                        else:
                            aps.append(0.0)
                    try:
                        boundary_acc = calculate_boundary_accuracy(thresholded_masks[j], masks[j])
                        if isinstance(boundary_acc, torch.Tensor):
                            boundary_acc = boundary_acc.detach().cpu().item()
                        else:
                            boundary_acc = float(boundary_acc)
                        boundary_accs.append(boundary_acc)
                    except Exception as e:
                        if i == 0:
                            print(f"Error calculating boundary accuracy: {e}")
                if visualize and vis_count < max_vis_samples:
                    num_to_add = min(batch_size, max_vis_samples - vis_count)
                    vis_images.extend(images[:num_to_add].detach().cpu())
                    vis_masks.extend(masks[:num_to_add].detach().cpu())
                    vis_preds.extend(thresholded_masks[:num_to_add].detach().cpu())
                    if 'prob_map' in outputs:
                        vis_probs.extend(outputs['prob_map'][:num_to_add].detach().cpu())
                    else:
                        vis_probs.extend(pred_masks[:num_to_add].detach().cpu())
                    if 'edge_map' in outputs:
                        vis_edges.extend(outputs['edge_map'][:num_to_add].detach().cpu())
                    print(f"Model output keys: {outputs.keys()}")
                    if 'threshold' in outputs:
                        batch_thresholds = outputs['threshold'][:num_to_add].detach().cpu()
                        print(f"Detected thresholds: shape={batch_thresholds.shape}, values={batch_thresholds}")
                        for j, thresh in enumerate(batch_thresholds):
                            prob_map = None
                            if 'prob_map' in outputs:
                                prob_map = outputs['prob_map'][j].detach().cpu()
                                if len(prob_map.shape) == 3 and prob_map.shape[0] == 1:
                                    prob_map = prob_map.squeeze(0)
                            if not isinstance(thresh, torch.Tensor) or thresh.numel() == 1:
                                thresh_value = float(thresh.item() if isinstance(thresh, torch.Tensor) else thresh)
                                print(f"Creating threshold map for sample {vis_count+j+1}, threshold value={thresh_value:.4f}")
                                if prob_map is not None:
                                    thresh_map = torch.zeros_like(prob_map)
                                    below_thresh = prob_map < thresh_value
                                    if below_thresh.any():
                                        thresh_map[below_thresh] = prob_map[below_thresh] / (2 * thresh_value + 1e-8)
                                    above_thresh = ~below_thresh
                                    if above_thresh.any():
                                        thresh_map[above_thresh] = 0.5 + (prob_map[above_thresh] - thresh_value) / (2 * (1 - thresh_value) + 1e-8)
                                    edge_width = 0.05
                                    near_thresh = torch.abs(prob_map - thresh_value) < edge_width
                                    if near_thresh.any():
                                        highlight = 0.7 * (1 - torch.abs(prob_map[near_thresh] - thresh_value) / edge_width)
                                        thresh_map[near_thresh] = torch.clamp(thresh_map[near_thresh] + highlight, 0, 1)
                                else:
                                    h, w = 256, 256
                                    thresh_map = torch.ones((h, w)) * thresh_value
                            else:
                                print(f"Sample {vis_count+j+1} using existing threshold map, shape={thresh.shape}")
                                thresh_map = thresh
                                if len(thresh_map.shape) == 3 and thresh_map.shape[0] == 1:
                                    thresh_map = thresh_map.squeeze(0)
                            if thresh_map.max() > 1.0 or thresh_map.min() < 0.0:
                                print(f"Warning: Threshold map values not in [0,1] range, will be clipped, min={thresh_map.min()}, max={thresh_map.max()}")
                                thresh_map = torch.clamp(thresh_map, 0.0, 1.0)
                            vis_thresh_maps.append(thresh_map)
                            print(f"Added threshold map, shape={thresh_map.shape}")
                    else:
                        print("Model doesn't output thresholds, creating threshold maps with default value 0.5")
                        for j in range(num_to_add):
                            prob_map = None
                            if 'prob_map' in outputs:
                                prob_map = outputs['prob_map'][j].detach().cpu()
                                if len(prob_map.shape) == 3 and prob_map.shape[0] == 1:
                                    prob_map = prob_map.squeeze(0)
                            if prob_map is not None:
                                thresh_value = 0.5
                                thresh_map = torch.zeros_like(prob_map)
                                below_thresh = prob_map < thresh_value
                                if below_thresh.any():
                                    thresh_map[below_thresh] = prob_map[below_thresh] / 1.0
                                above_thresh = ~below_thresh
                                if above_thresh.any():
                                    thresh_map[above_thresh] = prob_map[above_thresh]
                                edge_width = 0.05
                                near_thresh = torch.abs(prob_map - thresh_value) < edge_width
                                if near_thresh.any():
                                    highlight = 0.7 * (1 - torch.abs(prob_map[near_thresh] - thresh_value) / edge_width)
                                    thresh_map[near_thresh] = torch.clamp(thresh_map[near_thresh] + highlight, 0, 1)
                            else:
                                h, w = 256, 256
                                thresh_map = torch.ones((h, w)) * 0.5
                            thresh_map = torch.clamp(thresh_map, 0, 1)
                            vis_thresh_maps.append(thresh_map)
                            print(f"Added default threshold map, shape={thresh_map.shape}")
                    vis_count += num_to_add
            except Exception as e:
                print(f"Error processing batch {i+1}: {e}")
                continue
    mean_iou = np.mean(ious) if ious else 0.0
    mean_miou = np.mean(mious) if mious else 0.0
    mean_dice = np.mean(dices) if dices else 0.0
    mean_pa = np.mean(pas) if pas else 0.0
    mean_mpa = np.mean(mpas) if mpas else 0.0
    mean_ap = np.mean(aps) if aps else 0.0
    mean_boundary_acc = np.mean(boundary_accs) if boundary_accs else 0.0
    average_fps = total_frames / total_time if total_time > 0 else 0
    median_fps = np.median(fps_list) if fps_list else 0
    if visualize and vis_count > 0 and save_dir:
        try:
            vis_path = os.path.join(save_dir, 'visualization.png')
            if len(vis_edges) > 0:
                visualize_results(vis_images, vis_masks, vis_preds, vis_probs, vis_edges, vis_path)
            else:
                visualize_results(vis_images, vis_masks, vis_preds, vis_probs, None, vis_path)
            if True:
                try:
                    thresh_dir = os.path.join(save_dir, 'threshold_maps')
                    os.makedirs(thresh_dir, exist_ok=True)
                    for i, thresh_map in enumerate(vis_thresh_maps):
                        thresh_map_np = thresh_map.numpy()
                        if thresh_map_np.min() < 0 or thresh_map_np.max() > 1:
                            thresh_map_np = np.clip(thresh_map_np, 0, 1)
                        thresh_map_uint8 = (thresh_map_np * 255).astype(np.uint8)
                        if len(thresh_map_uint8.shape) > 2:
                            thresh_map_uint8 = thresh_map_uint8.squeeze()
                        print(f"Threshold map shape: {thresh_map_uint8.shape}, type: {thresh_map_uint8.dtype}, range: [{thresh_map_uint8.min()}, {thresh_map_uint8.max()}]")
                        try:
                            colored_thresh = cv2.applyColorMap(thresh_map_uint8, cv2.COLORMAP_VIRIDIS)
                            thresh_path = os.path.join(thresh_dir, f'threshold_map_sample{i+1}.png')
                            cv2.imwrite(thresh_path, colored_thresh)
                            print(f"Threshold map saved to {thresh_path}")
                            sample_dir = os.path.join(save_dir, f'sample{i+1}')
                            if os.path.exists(sample_dir):
                                try:
                                    sample_thresh_path = os.path.join(sample_dir, "threshold_map.png")
                                    cv2.imwrite(sample_thresh_path, colored_thresh)
                                    print(f"Threshold map saved to {sample_thresh_path}")
                                except Exception as e:
                                    print(f"Failed to save to sample directory: {e}")
                        except Exception as e:
                            print(f"Failed to apply color mapping: {e}")
                            thresh_path = os.path.join(thresh_dir, f'threshold_map_sample{i+1}_gray.png')
                            cv2.imwrite(thresh_path, thresh_map_uint8)
                            print(f"Grayscale threshold map saved to {thresh_path}")
                    print(f"Threshold maps saved to: {thresh_dir} and respective sample directories")
                except Exception as e:
                    print(f"Error generating threshold maps: {e}")
        except Exception as e:
            print(f"Error creating visualization results: {e}")
    print("\n" + "="*50)
    print("Test Results:")
    print(f"IoU: {mean_iou:.4f}")
    print(f"mIoU: {mean_miou:.4f}")
    print(f"Dice Coefficient: {mean_dice:.4f}")
    print(f"Pixel Accuracy (PA): {mean_pa:.4f}")
    print(f"Mean Pixel Accuracy (mPA): {mean_mpa:.4f}")
    print(f"Average Precision (AP): {mean_ap:.4f}")
    print(f"Boundary Accuracy: {mean_boundary_acc:.4f}")
    print("-"*50)
    print(f"Performance Analysis:")
    print(f"Average FPS: {average_fps:.2f}")
    print(f"Median FPS: {median_fps:.2f}")
    print(f"Total Inference Time: {total_time:.2f} seconds")
    print("="*50)
    return {
        'iou': mean_iou,
        'miou': mean_miou,
        'dice': mean_dice,
        'pa': mean_pa,
        'mpa': mean_mpa,
        'ap': mean_ap,
        'boundary_acc': mean_boundary_acc,
        'avg_fps': average_fps,
        'median_fps': median_fps
    }
def main():
    if not check_dependencies():
        print("Missing necessary dependencies, cannot continue testing")
        return
    parser = argparse.ArgumentParser(description='Test wood defect segmentation model')
    parser.add_argument('--model_path', type=str, help='Model path')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--visualize', action='store_true', default=True, help='Whether to visualize results')
    args = parser.parse_args()
    model_path = args.model_path
    if model_path is None:
        model_path = input("Please enter the model file path (default is outputs/best_model.pth): ").strip()
        if not model_path:
            model_path = "outputs/best_model.pth"
    config = Config()
    if not hasattr(config, 'test_image_dir'):
        config.test_image_dir = os.path.join(config.data_dir, 'test/images')
        config.test_mask_dir = os.path.join(config.data_dir, 'test/masks')
    if not os.path.exists(config.test_image_dir):
        print("Warning: Test set directory doesn't exist, using validation set instead")
        config.test_image_dir = config.val_image_dir
        config.test_mask_dir = config.val_mask_dir
    save_dir = os.path.join(config.save_dir, 'predictions')
    os.makedirs(save_dir, exist_ok=True)
    try:
        test_dataset = WoodDefectDataset(
            config.test_image_dir,
            config.test_mask_dir,
            is_train=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        print(f"Loaded {len(test_dataset)} test samples")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        model = WoodDefectBD(pretrained=False)
        print(f"Attempting to load model: {model_path}")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("Model loaded successfully from checkpoint dictionary")
            else:
                model.load_state_dict(checkpoint)
                print("Model loaded successfully")
        else:
            print(f"Warning: Model file {model_path} doesn't exist, using randomly initialized weights")
        model = model.to(device)
        print("Model loaded to device, starting testing...")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    try:
        results = test(model, test_loader, device, save_dir, visualize=args.visualize, max_vis_samples=3)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        result_file = os.path.join(config.save_dir, f'test_results_{model_name}.txt')
        with open(result_file, 'w') as f:
            f.write("Test Results\n")
            f.write(f"IoU: {results['iou']:.4f}\n")
            f.write(f"mIoU: {results['miou']:.4f}\n")
            f.write(f"Dice Coefficient: {results['dice']:.4f}\n")
            f.write(f"Pixel Accuracy (PA): {results['pa']:.4f}\n")
            f.write(f"Mean Pixel Accuracy (mPA): {results['mpa']:.4f}\n")
            f.write(f"Average Precision (AP): {results['ap']:.4f}\n")
            f.write(f"Boundary Accuracy: {results['boundary_acc']:.4f}\n")
            f.write("-"*50 + "\n")
            f.write(f"Average FPS: {results['avg_fps']:.2f}\n")
            f.write(f"Median FPS: {results['median_fps']:.2f}\n")
        print(f"Test results saved to: {result_file}")
        return results
    except Exception as e:
        print(f"Error during testing: {e}")
        return None
if __name__ == '__main__':
    result = main()
    if result:
        print("\nTesting completed!")