import cv2
import numpy as np
import os
import torch

from predict.pred_utils import get_imgs, looking_for_event_index_by_timestamp, psnr_, ssim_


indexing_skip_ind = {
    'basket_09':[31, 32, 33, 34],
    'may29_rooftop_handheld_02':[17, 70],
    'may29_rooftop_handheld_03':[306],
    'may29_rooftop_handheld_05':[21],
}


def predict_bsergb(model, num_bins, device, save_path, multis=[1,3,5], isSave=False, isTestPer=False, 
                  saveSpecificScene=None):
    print("Start test BSERGB!")
    test_path = '/DATASSD1/BSERGB/1_TEST'
    for multi in multis:
        psnr_mul = []
        ssim_mul = []
        for scene in os.listdir(test_path):
            if scene != "":
                psnr_all = []
                ssim_all = []
                val_folder = os.path.join(save_path, scene + '_' + str(multi))
                img_folder = os.path.join(test_path, scene, 'images')
                img_names = os.listdir(img_folder)
                img_names.sort()
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                if not os.path.exists(val_folder):
                    os.mkdir(val_folder)
                img_paths = [os.path.join(img_folder, i) for i in img_names if i.endswith("png") or i.endswith("jpg")]
                img_paths.sort()
                h, w, _ = cv2.imread(img_paths[0]).shape
                index = 0
                while (index + multi + 3) < len(img_names):
                    imgs = get_imgs(img_paths, multi, index)
                    for i in range(multi):
                        # if index + i + 1 < 30:
                        #     continue
                        gt = cv2.imread(img_paths[index + i + 1])
                        img_index0, gt_index, img_index1 = index, index + i + 1, index + multi + 1
                        
                        skip_flag = False
                        if scene in indexing_skip_ind:
                            for skip_index in indexing_skip_ind[scene]:
                                if img_index0 <= skip_index < img_index1:
                                    skip_flag = True
                                    break
                        if skip_flag:
                            continue
                        
                        event_voxel, mask = looking_for_event_index_by_timestamp(
                            os.path.join(test_path, scene), num_bins,
                            img_index0, gt_index,
                            img_index1, h, w, 240, real=True, bsergb=True)
                        
                        event_voxel = event_voxel.to(device, non_blocking=True)
                        mask = mask.to(device, non_blocking=True)
                        with torch.no_grad():
                            pred = model.inference(imgs, event_voxel, mask)
                        pred_out = (pred[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                        psnr = psnr_(gt, pred_out)
                        ssim = ssim_(gt, pred_out)
                        psnr_all.append(psnr)
                        psnr_mul.append(psnr)
                        ssim_all.append(ssim)
                        ssim_mul.append(ssim)
                        if isTestPer:
                            print(scene, "\'s ", gt_index, "\'s PSNR is ", np.array(psnr_all).mean(), " SSIM is ", np.array(ssim_all).mean())
                            
                        save_name = os.path.join(val_folder, img_names[index + i + 1])
                        if isSave:
                            if isinstance(saveSpecificScene, list):
                                if scene in saveSpecificScene:
                                    cv2.imwrite(save_name, pred_out)
                            else:
                                cv2.imwrite(save_name, pred_out)
                    index = index + multi
                print(scene, "\'s ", multi, "\'s PSNR is ", np.array(psnr_all).mean())
                print(scene, "\'s ", multi, "\'s SSIM is ", np.array(ssim_all).mean())
        print(multi, "\'s PSNR is ", np.array(psnr_mul).mean())
        print(multi, "\'s SSIM is ", np.array(ssim_mul).mean())
