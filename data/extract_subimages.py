"""A multi-thread tool to crop large images to sub-images for faster IO.
   (This preprocessing code is copied and modified from official implement:
    https://github.com/open-mmlab/mmsr/tree/master/codes/data_scripts)"""
import os
import os.path as osp
import sys
import glob
from multiprocessing import Pool
import numpy as np
import cv2
from PIL import Image
import time
from shutil import get_terminal_size


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def imgs_divide(HR_train_folder,LR_infos,crop_sz=480,step=240,theres_sz=48):
    opt = {}
    opt['n_thread'] = 10
    opt['compression_level'] = 3
    # process HR images
    print('process HR images')
    opt['input_folder'] = HR_train_folder
    opt['save_folder'] = os.path.join(os.path.dirname(HR_train_folder),"sub",os.path.basename(HR_train_folder)+"_sub")
    opt['crop_sz'] = crop_sz  # the size of each sub-image
    opt['step'] = step  # step of the sliding crop window
    opt['thres_sz'] = theres_sz  # size threshold
    extract_signle(opt)
    # process LR images
    for index,lr_info in enumerate(LR_infos,1):
        print('[{}/{}]process {}'.format(index,len(LR_infos),lr_info['folder']))
        opt['input_folder'] = lr_info['folder']
        # calculate save path, exp. './data/DIV2K/sub/DIV2K_train_LR_bicubic_X2_sub'
        base_dir = os.path.dirname(os.path.dirname(lr_info['folder']))
        scale_name = os.path.basename(lr_info['folder'])
        type_name = os.path.basename(os.path.dirname(lr_info['folder']))
        save_folder = os.path.join(base_dir,"sub",type_name+"_"+scale_name+"_sub")
        scale_ratio = lr_info['scale']
        opt['save_folder'] = save_folder
        opt['crop_sz'] = crop_sz // scale_ratio
        opt['step'] = step // scale_ratio
        opt['thres_sz'] = theres_sz // scale_ratio
        extract_signle(opt)


def extract_signle(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    img_list = _get_paths_from_images(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    img_name = osp.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            cv2.imwrite(
                osp.join(opt['save_folder'],
                         img_name.replace('.png', '_s{:03d}.png'.format(index))), crop_img,
                [cv2.IMWRITE_PNG_COMPRESSION, opt['compression_level']])
    return 'Processing {:s} ...'.format(img_name)


# ##############
# ### Utils ####
# ##############
class ProgressBar(object):
    '''A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    '''

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time + 1e-9
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


# ###################
# ### Data Utils ####
# ###################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images

def rename(sub_dir):
    '''delete the fix labes (x2,3,x4,x8) of image name in LR folder'''
    def DIV2K(path):
        img_path_l = glob.glob(os.path.join(path, '*'))
        for img_path in img_path_l:
            new_path = img_path.replace('x2', '').replace('x3', '').replace(
                'x4', '').replace('x8', '')
            os.rename(img_path, new_path)

    def _batch_rename(sub_dir):
        folders = [os.path.join(sub_dir, n) for n in os.listdir(sub_dir)]
        LR_folders = [folder for folder in folders if 'HR' not in os.path.basename(folder)]
        for index, folder in enumerate(LR_folders, 1):
            print(f"[{index}/{len(folders)}] Renaming in {folder}")
            DIV2K(folder)
        print('Done')
    _batch_rename(sub_dir)

def main():
    # process train set
    HR_train_folder = './data/DIV2K/DIV2K_train_HR'
    LR_train_bicubic_X2 = {"folder": r'./data/DIV2K/DIV2K_train_LR_bicubic/X2', "scale": 2}
    LR_train_bicubic_X3 = {"folder": r'./data/DIV2K/DIV2K_train_LR_bicubic/X3', "scale": 3}
    LR_train_bicubic_X4 = {"folder": r'./data/DIV2K/DIV2K_train_LR_bicubic/X4', "scale": 4}
    LR_train_unknown_X2 = {"folder": r'./data/DIV2K/DIV2K_train_LR_unknown/X2', "scale": 2}
    LR_train_unknown_X3 = {"folder": r'./data/DIV2K/DIV2K_train_LR_unknown/X3', "scale": 3}
    LR_train_unknown_X4 = {"folder": r'./data/DIV2K/DIV2K_train_LR_unknown/X4', "scale": 4}
    LR_infos = [LR_train_bicubic_X2, LR_train_bicubic_X3, LR_train_bicubic_X4, LR_train_unknown_X2, LR_train_unknown_X3,
                LR_train_unknown_X4]
    imgs_divide(HR_train_folder, LR_infos, crop_sz=480, step=240, theres_sz=48)
    # rename
    sub_dir = './data/DIV2K/sub/'
    rename(sub_dir)



if __name__ == '__main__':
    main()
