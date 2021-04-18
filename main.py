from data import extract_subimages,png2tfrecord
from track1_bicubic import train_esrgan_ea_bicubic
from track1_bicubic import train_esrgan_bicubic
from track1_bicubic.result import test as test_bicubic
from track2_unknown import train_esrgan_ea_unknown
from track2_unknown import train_esrgan_unknown
from track2_unknown.result import test as test_unknown
# ======================================================================================================================
# Data preprocessing (need DIV2K images available in the folder "./data/DIV2K")
extract_subimages.main() # Crop images
png2tfrecord.convert() # Generate tfrecord files for training
# ======================================================================================================================
gpu = '0' # assgin the gpu to use
# ======================================================================================================================
# Track 1 : bicubic downgraded tasks

## bicubic X2
# train ESRGAN
train_esrgan_bicubic.train(gpu,2)
# train ESRGAN-EA
train_esrgan_ea_bicubic.train(gpu,2)
# test ESRGAN
test_bicubic(gpu,'esrgan',2)
# test ESRGAN-EA
test_bicubic(gpu,'esrgan_ea',2)

## bicubic X4
# train ESRGAN
train_esrgan_bicubic.train(gpu,3)
# train ESRGAN-EA
train_esrgan_ea_bicubic.train(gpu,3)
# test ESRGAN
test_bicubic(gpu,'esrgan',3)
# test ESRGAN-EA
test_bicubic(gpu,'esrgan_ea',3)

## bicubic X4
# train ESRGAN
train_esrgan_bicubic.train(gpu,4)
# train ESRGAN-EA
train_esrgan_ea_bicubic.train(gpu,4)
# test ESRGAN
test_bicubic(gpu,'esrgan',4)
# test ESRGAN-EA
test_bicubic(gpu,'esrgan_ea',4)


# ======================================================================================================================
# Track 2 : unknown downgraded tasks

## unknown X2
# train ESRGAN
train_esrgan_unknown.train(gpu,2)
# train ESRGAN-EA
train_esrgan_ea_unknown.train(gpu,2)
# test ESRGAN
test_unknown(gpu,'esrgan',2)
# test ESRGAN-EA
test_unknown(gpu,'esrgan_ea',2)

## unknown X3
# train ESRGAN
train_esrgan_unknown.train(gpu,2)
# train ESRGAN-EA
train_esrgan_ea_unknown.train(gpu,2)
# test ESRGAN
test_unknown(gpu,'esrgan',2)
# test ESRGAN-EA
test_unknown(gpu,'esrgan_ea',2)

## unknown X4
# train ESRGAN
train_esrgan_unknown.train(gpu,2)
# train ESRGAN-EA
train_esrgan_ea_unknown.train(gpu,2)
# test ESRGAN
test_unknown(gpu,'esrgan',2)
# test ESRGAN-EA
test_unknown(gpu,'esrgan_ea',2)
