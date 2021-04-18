#from absl import app, flags, logging
from absl.flags import FLAGS
import os
import tqdm
import glob
import random
import tensorflow as tf




def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example_bin(img_name, hr_img_str, lr_img_str):
    # Create a dictionary with features that may be relevant (binary).
    feature = {'image/img_name': _bytes_feature(img_name),
               'image/hr_encoded': _bytes_feature(hr_img_str),
               'image/lr_encoded': _bytes_feature(lr_img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def make_example(img_name, hr_img_path, lr_img_path):
    # Create a dictionary with features that may be relevant.
    feature = {'image/img_name': _bytes_feature(img_name),
               'image/hr_img_path': _bytes_feature(hr_img_path),
               'image/lr_img_path': _bytes_feature(lr_img_path)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(lr_dataset_path,hr_dataset_path,output_filename):
    samples = []
    print('Reading data list...')
    for hr_img_path in glob.glob(os.path.join(hr_dataset_path, '*.png')):
        img_name = os.path.basename(hr_img_path).replace('.png', '')
        lr_img_path = os.path.join(lr_dataset_path, img_name + '.png')
        samples.append((img_name, hr_img_path, lr_img_path))
    random.shuffle(samples)

    if os.path.exists(output_filename):
        print('{:s} already exists. Exit...'.format(
            output_filename))
        exit(1)

    print('Writing {} sample to tfrecord file...'.format(len(samples)))

    print(output_filename)
    with tf.io.TFRecordWriter(output_filename) as writer:
        for img_name, hr_img_path, lr_img_path in tqdm.tqdm(samples):
            # binary
            # hr_img_str = open(hr_img_path, 'rb').read()
            # lr_img_str = open(lr_img_path, 'rb').read()
            # tf_example = make_example_bin(img_name=str.encode(img_name),
            #                               hr_img_str=hr_img_str,
            #                               lr_img_str=lr_img_str)

            # not binary
            lr_img_path = os.path.abspath(lr_img_path)
            hr_img_path = os.path.abspath(hr_img_path)


            tf_example = make_example(img_name=str.encode(img_name),
                                      hr_img_path=str.encode(hr_img_path),
                                      lr_img_path=str.encode(lr_img_path))
            writer.write(tf_example.SerializeToString())






def main():
    sub_dir = './data/DIV2K/sub'
    folders = [os.path.join(sub_dir,n) for n in os.listdir(sub_dir)]
    print(folders)
    lr_folders = []
    print(folders)
    for folder in folders:
        if 'HR' in os.path.basename(folder):
            hr_folder = folder
        elif 'LR' in os.path.basename(folder):
            lr_folders.append(folder)

    for index,lr_folder in enumerate(lr_folders):
        print(f'processing {index}/{len(lr_folders)}')
        name = os.path.basename(lr_folder)
        name += '.tfrecord'
        output_dir = './data'
        output_filename = os.path.join(output_dir,name)
        write_tfrecord(lr_folder, hr_folder, output_filename)
    print('Done')

def single(lr_sub):
    sub_dir = './data/DIV2K/sub'
    hr_folder = os.path.join(sub_dir,'DIV2K_train_HR_sub')
    lr_folder = os.path.join(sub_dir,lr_sub)
    name = os.path.basename(lr_folder)
    name += '.tfrecord'
    output_dir = './data'
    output_filename = os.path.join(output_dir, name)
    write_tfrecord(lr_folder, hr_folder, output_filename)

def convert():
    main()

if __name__ == '__main__':
    pass
    # try:
    #     app.run(main)
    # except SystemExit:
    #     pass
    #single('DIV2K_train_LR_bicubic_X3_sub')