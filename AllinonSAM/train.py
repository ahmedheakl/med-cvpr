import torch

import sys
import copy
import os

from data_utils import *
from model import *
from utils import *
import yaml
from tqdm import tqdm
import wandb
def print_model_parameters_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")

    # Print parameters by module
    print("\nParameters by module:")
    for name, module in model.named_children():
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print("*************************************************************************************************************")
        print(f"  {name}:")
        print(f"    Total: {total_params:,}")
        print(f"    Trainable: {trainable_params:,}")
        print(f"    Frozen: {frozen_params:,}")
        print("*******************************************************************************************")

def train(model, tr_dataset, val_dataset, criterion, optimizer, sav_path='./checkpoints/temp.pth', num_epochs=25, bs=32, device='cuda:0'):
    model = model.to(device)
    best_loss = 100000.0
    best_dice = 0
    best_HD95=1000000.0
    print("Training parameters: \n----------")
    print("batch size: ", bs)
    print("num epochs: ", num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        bs_count=0
        inputs_li, labels_li, text_ids_li, text_li, slice_num_li = [], [], [], [], []
        running_loss = 0
        running_dice = 0
        count = 0
        #run training
        # print("eere: ",len(tr_dataset))
        for i in range(len(tr_dataset)):
            inputs, labels,_, text, slice_nums = tr_dataset[i]
            inputs_li.append(inputs)
            labels_li.append(labels)
            text_li = text_li + [text]*(inputs.shape[0])
            slice_num_li = slice_num_li + slice_nums
            bs_count += 1
            if (bs_count%bs==0) or (i==len(tr_dataset)-1):
                #start training
                bs_count=0
                inputs = torch.cat(inputs_li,dim=0)
                labels = torch.cat(labels_li, dim=0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    outputs, reg_loss = model(inputs, text_li, slice_num_li)
                    seg_loss=0
                    for c in criterion:
                        seg_loss += c(outputs, labels.float())
                    seg_loss.backward()
                    optimizer.step()
                    running_loss += seg_loss.cpu()
                
                preds = (outputs>=0.5)
                ri, ru = running_stats(labels,preds)
                running_dice += dice_collated(ri,ru)
                count += ri.shape[0]
                
                inputs_li = []
                labels_li = []
                text_li = []
                slice_num_li = []
        epoch_dice = running_dice / count
        
        print("Training loss: ", running_loss/(1+(len(tr_dataset)//bs)))
        print("Training dice: ", epoch_dice)

        #do val if epoch is a multiple of 5
        if epoch%5==0:
            running_dice = 0
            count=0
            for i in range(len(val_dataset)):
                inputs, labels,_, text, slice_nums = val_dataset[i]
                inputs_li.append(inputs)
                labels_li.append(labels)
                text_li = text_li + [text]*(inputs.shape[0])
                slice_num_li = slice_num_li + slice_nums
                bs_count += 1
                if bs_count%bs==0:
                    #start training
                    bs_count=0
                    inputs = torch.cat(inputs_li,dim=0)
                    labels = torch.cat(labels_li, dim=0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with torch.set_grad_enabled(False):
                        outputs, reg_loss = model(inputs, text_li, slice_num_li)
                        preds = (outputs>=0.5)
                        ri, ru = running_stats(labels,preds)
                        running_dice += dice_collated(ri,ru)
                        count += ri.shape[0]

                    inputs_li = []
                    labels_li = []
                    text_li = []
                    slice_num_li = []
            # epoch_dice = running_dice / (len(val_dataset))
            epoch_dice = running_dice / count

            print(f'Val Dice: {epoch_dice:.4f}')            

            # deep copy the model
            if epoch_dice > best_dice:
                # best_loss = epoch_loss
                best_dice = epoch_dice
                torch.save(model.state_dict(),sav_path)

    return model



import torch
import wandb
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from utils import running_stats, dice_collated, compute_hd95,fractal_dimension
from huggingface_hub import HfApi
import os
from PIL import Image
import numpy as np
from pathlib import Path

# Load configuration from data_config.yml
with open('/l/users/sarim.hashmi/cvpr/med-cvpr/AllinonSAM/config_arcade.yml', 'r') as data_config_file:
    data_config = yaml.safe_load(data_config_file)

# Load configuration from model_svdtuning.yml
with open('/l/users/sarim.hashmi/cvpr/med-cvpr/AllinonSAM/model_lora_decoder_encoder.yml', 'r') as model_config_file:
    model_config = yaml.safe_load(model_config_file)
def train_dl(model, datasets, dataset_sizes, criterion, optimizer, scheduler, 
             sav_path='./checkpoints/temp.pth', num_epochs=25, bs=32, 
             device='cuda:0', retain_graph=False, neg2pos_ratio=-1, 
             save_dir="./validation_images", reg_multiplier=0.01,
             push_to_hub=True, hub_model_id=None, hub_token=None):
    """
    Training function with Hugging Face Hub integration
    Additional params:
    push_to_hub: bool, whether to push model to Hub
    hub_model_id: str, format: "username/model-name"
    hub_token: str, Hugging Face access token
    """
    torch.cuda.empty_cache()
    model = model.to(device)
    best_dice = 0
    best_loss = 10000
    best_hd95 = 1000000
    print_model_parameters_stats(model)

    # Create directories for saving images
    print(save_dir)
    label_dir = os.path.join(save_dir, 'labels')
    pred_dir = os.path.join(save_dir, 'pred_labels')
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    run_name = f"{data_config['data']['root_path'].split('/')[-1]}"
    # Initialize wandb
    wandb.init(project="final_bench", name='drive_unfreeze_lora_rank8', config={
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": bs,
        "num_epochs": num_epochs,
        "reg_multiplier": reg_multiplier
    })

    print("Training parameters: \n----------")
    print('number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("batch size: ", bs)
    print("num epochs: ", num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        dataloaders = {}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                if neg2pos_ratio > 0:
                    datasets[phase].generate_examples(neg2pos_ratio)
            else:
                model.eval()

            running_loss = 0.0
            running_hd95 = 0.0 
            all_preds = []
            running_dice = 0
            count = 0
            dataloaders[phase] = torch.utils.data.DataLoader(datasets[phase], batch_size=bs, shuffle=True, num_workers=4)

            pbar = tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=False)

            for batch_idx, (inputs, labels, text_idxs, text) in enumerate(pbar):
                count += 1
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs, reg_loss = model(inputs, text)
                    if len(outputs.shape) == 4:
                        outputs = torch.squeeze(outputs, dim=1)
                    loss = 0
                    seg_loss = 0
                    for c in criterion:
                        if 'text' in c.__code__.co_varnames:
                            seg_loss += c(outputs, text, labels.float())
                        else:
                            seg_loss += c(outputs, labels.float())
                    loss += seg_loss
                    loss += (reg_loss * reg_multiplier)

                    if phase == 'train':
                        loss.backward(retain_graph=True)
                        optimizer.step()

                with torch.no_grad():
                    preds = (outputs >= 0.5)
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels, preds)
                    running_dice += dice_collated(ri, ru)
                    hd95 = compute_hd95(preds, labels)
                    running_hd95 += hd95.item()

                    if phase == 'val':
                        all_preds.append(preds.cpu().numpy())

                    if phase == 'val' and epoch % 10 == 0 and batch_idx < 5:
                        for i in range(min(2, inputs.size(0))):
                            img_name = f"epoch_{epoch}_batch_{batch_idx}_img_{i}.png"
                            
                            label_img = labels[i].cpu().numpy() * 255
                            label_img = Image.fromarray(label_img.astype(np.uint8))
                            label_img.save(os.path.join(label_dir, img_name))
                            
                            pred_img = preds[i].cpu().numpy() * 255
                            pred_img = Image.fromarray(pred_img.astype(np.uint8))
                            pred_img.save(os.path.join(pred_dir, img_name))

                pbar.set_postfix({'loss': loss.item(), 'dice': running_dice / count, "hd95": running_hd95 / count})

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_dice = running_dice / dataset_sizes[phase]
            epoch_hd95 = running_hd95 / dataset_sizes[phase]

            if phase == 'val':
                all_preds = np.concatenate(all_preds, axis=0) 
                fractal_dim_values = []
                for i in range(all_preds.shape[0]):
                    fractal_dim = fractal_dimension(all_preds[i])
                    fractal_dim_values.append(fractal_dim)

                average_fractal_dim = np.mean(fractal_dim_values)
                wandb.log({"average_fractal_dimension": average_fractal_dim})

            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}  HD95: {epoch_hd95:.4f}')

            wandb.log({f"{phase}_loss": epoch_loss, f"{phase}_hd95": epoch_hd95, 
                      f"{phase}_dice": epoch_dice, "epoch": epoch})

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_dice = epoch_dice
                best_hd95 = epoch_hd95
                wandb.run.summary["best_val_loss"] = best_loss
                wandb.run.summary["best_val_dice"] = best_dice
                wandb.run.summary["best_val_hd95"] = best_hd95
            
            elif phase == 'val' and np.isnan(epoch_loss):
                print("nan loss but saving model")
                torch.save(model.state_dict(), sav_path)

    print(f'Best val loss: {best_loss:4f}, best val dice: {best_dice:2f}, best Hd95: {best_hd95:2f}')
    
    # Save the final model
    model_save_path = f"{sav_path}/final_model.pth"
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved at: {model_save_path}")

    # Push to Hugging Face Hub if requested
    if push_to_hub and hub_model_id and hub_token:
        try:
            # Save model config
            config_save_path = os.path.join(model_dir, "config.json")
            model_config = {
                "architecture": "lora rank 8 unfreeze drive",
                "best_val_loss": float(best_loss),
                "best_val_dice": float(best_dice),
                "best_val_hd95": float(best_hd95),
                "training_params": {
                    "batch_size": bs,
                    "num_epochs": num_epochs,
                    "reg_multiplier": reg_multiplier
                }
            }
            
            with open(config_save_path, 'w') as f:
                json.dump(model_config, f, indent=2)

            # Initialize Hugging Face API
            api = HfApi()
            
            # Login
            api.set_access_token(hub_token)
            
            # Create repository if it doesn't exist
            try:
                api.create_repo(repo_id=hub_model_id, exist_ok=True)
            except Exception as e:
                print(f"Repository creation warning (might already exist): {e}")

            # Push files to hub
            api.upload_file(
                path_or_fileobj=model_save_path,
                path_in_repo="final_model.pth",
                repo_id=hub_model_id,
                commit_message="Upload trained model weights"
            )
            
            api.upload_file(
                path_or_fileobj=config_save_path,
                path_in_repo="config.json",
                repo_id=hub_model_id,
                commit_message="Upload model config"
            )
            
            print(f"Successfully pushed model to Hugging Face Hub: {hub_model_id}")
            
        except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {e}")
    
    # Finish wandb run
    wandb.finish()

    return model
