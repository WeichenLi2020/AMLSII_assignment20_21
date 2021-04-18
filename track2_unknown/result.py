from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from track2_unknown.modules.models import RRDB_Model,RRDB_Model_scale
from track2_unknown.modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim,calculate_edge_loss)


def main(gpu,yaml):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()
    yaml_path = os.path.join(yaml)
    cfg = load_yaml(yaml_path)
    # define network
    def model_generator(cfg):
        model_type = cfg['sub_name']
        type1 = {'esrgan_X2_unknown': 2, 'esrgan_X3_unknown': 3, 'esrgan_X4_unknown': 4}
        type2 = {'esrgan_ea_X2_unknown': 2, 'esrgan_ea_X3_unknown': 3, 'esrgan_ea_X4_unknown': 2}
        if model_type in type1:
            model = RRDB_Model(None, cfg['ch_size'], cfg['network_G'], scale=cfg['scale'])
        elif model_type in type2:
            model = RRDB_Model_scale(None, cfg['ch_size'], cfg['network_G'], scale=cfg['scale'])
        return model

    model = model_generator(cfg)
    # load checkpoint
    checkpoint_dir = './track2_unknown/checkpoints/'+ cfg['sub_name']
    checkpoint = tf.train.Checkpoint(model=model)
    if tf.train.latest_checkpoint(checkpoint_dir):
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        print("[*] load ckpt from {}.".format(
            tf.train.latest_checkpoint(checkpoint_dir)))
    else:
        print("[*] Cannot find ckpt from {}.".format(checkpoint_dir))
        exit()

    # process
    results_path = './results/' + cfg['sub_name'] + '/'
    print("[*] Processing on validation set and write results")

    HR_path = None
    for key, path in cfg['test_dataset'].items():
        # save ground truth images path
        if 'HR' in key:
            if not HR_path:
                HR_path = path
                continue
            else:
                print('[*] Multiple HR sets found, please avoid key word \'HR\' in LR set name')
                return
        # process LR images
        print("'{}' from {}\n  PSNR/SSIM".format(key, path))
        dataset_name = key.replace('_path', '')
        pathlib.Path(results_path + dataset_name).mkdir(
            parents=True, exist_ok=True)
        # read hr set and save results
        f_detail = open(os.path.join(results_path,dataset_name,'details.txt'),mode='w')
        f_detail.write("picture_id\tPSNR\tSSIM\tEDGE_LOSS\n")
        f_summary = open(os.path.join(results_path,dataset_name,'summary.txt'),mode='w')
        f_summary.write("avg_PSNR\tavg_SSIM\tavg_EDGE_LOSS\n")

        # calculate the averge of PSNR,SSIM,EDGE LOSS
        num_item=0
        avg_PSNR = avg_SSIM = avg_EDGELOSS=0
        for img_name in os.listdir(path):
            lr_img = cv2.imread(os.path.join(path, img_name))
            hr_img = cv2.imread(
                os.path.join(HR_path, img_name.replace('x2', '').replace('x3', '').replace('x4', '').replace('x8', '')))

            sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
            bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)
            # calculate PSNR,SSIM,EDGE_LOSS of bicubic images
            PSNR_bic = calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img))
            SSIM_bic = calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img))
            EDGELOSS_bic = calculate_edge_loss(bic_img, hr_img)
            # calculate PSNR,SSIM,EDGE_LOSS of SR images
            num_item+=1
            PSNR_SR = calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))
            SSIM_SR = calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))
            EDGELOSS_SR =calculate_edge_loss(sr_img, hr_img)
            # write details
            f_detail.write(f"{img_name}\t{PSNR_SR}\t{SSIM_SR}\t{EDGELOSS_SR}\n")
            f_detail.flush()

            avg_PSNR += PSNR_SR
            avg_SSIM += SSIM_SR
            avg_EDGELOSS += EDGELOSS_SR
            str_format = "  [{}] Bic={:.2f}db/{:.2f}/{:.2f}, SR={:.2f}db/{:.2f}/{:.2f}"
            print(str_format.format(
                img_name + ' ' * max(0, 20 - len(img_name)),
                PSNR_bic,
                SSIM_bic,
                EDGELOSS_bic,
                PSNR_SR,
                SSIM_SR,
                EDGELOSS_SR, ))
            result_img_path = os.path.join(
                results_path + dataset_name, 'Bic_SR_HR_' + img_name)
            results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
            cv2.imwrite(result_img_path, results_img)
        # write summary
        f_summary.write(f"{avg_PSNR/num_item}\t{avg_SSIM/num_item}\t{avg_EDGELOSS/num_item}\n")
    print("[*] write the visual results in {}".format(results_path))

def test(gpu,model,scale):
    # config files dictionar
    esrgan_ea_yaml_dicts = {2:'./track2_unknown/configs/esrgan_ea_x2_unknown.yaml',
                  3:'./track2_unknown/configs/esrgan_ea_x3_unknown.yaml',
                  4:'./track2_unknown/configs/esrgan_ea_x4_unknown.yaml'}
    esrgan_yaml_dicts = {2: './track2_unknown/configs/esrgan_x2_unknown.yaml',
                            3: './track2_unknown/configs/esrgan_x3_unknown.yaml',
                            4: './track2_unknown/configs/esrgan_x4_unknown.yaml'}
    models = {'esrgan':esrgan_yaml_dicts,'esrgan_ea':esrgan_ea_yaml_dicts}
    # check availability and run test
    if(model in models and scale in models[model]):
        main(gpu,models[model][scale])
    else:
        print('Only support esrgan and esrgan_ea in scale 2, scale 3 and scale 4.')
