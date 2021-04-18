from absl import app, flags, logging
from absl.flags import FLAGS
import os
import sys
import tensorflow as tf
from track2_unknown.modules.models import RRDB_Model, DiscriminatorVGG128,RRDB_Model_scale
from track2_unknown.modules.lr_scheduler import MultiStepLR
from track2_unknown.modules.losses import (PixelLoss, ContentLoss, EdgeLoss,DiscriminatorLoss,
                            GeneratorLoss)
from track2_unknown.modules.utils import (load_yaml, load_dataset, ProgressBar,
                           set_memory_growth)

# flags.DEFINE_string('cfg_path', './configs/esrgan_ea_x4_bicubic.yaml', 'config file path')
# flags.DEFINE_string('gpu', '7', 'which gpu to use')


def main(gpu,yaml_path):
    # init
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu  # which GPU to use

    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    cfg = load_yaml(yaml_path)

    # define network
    generator = RRDB_Model_scale(cfg['input_size'], cfg['ch_size'], cfg['network_G'],scale=cfg['scale'])
    generator.summary(line_length=80)
    discriminator = DiscriminatorVGG128(cfg['gt_size'], cfg['ch_size'])
    discriminator.summary(line_length=80)

    # extration_variables = [var for var in generator.trainable_variables if 'upsample' not in var.name]
    # upsample_variables = [var for var in generator.trainable_variables if 'upsample' in var.name]

    train_dataset = load_dataset(cfg, 'train_dataset', shuffle=False)

    # define optimizer
    learning_rate_G = MultiStepLR(cfg['lr_G'], cfg['lr_steps'], cfg['lr_rate'])
    learning_rate_D = MultiStepLR(cfg['lr_D'], cfg['lr_steps'], cfg['lr_rate'])
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=learning_rate_G,
                                           beta_1=cfg['adam_beta1_G'],
                                           beta_2=cfg['adam_beta2_G'])
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=learning_rate_D,
                                           beta_1=cfg['adam_beta1_D'],
                                           beta_2=cfg['adam_beta2_D'])



    # define losses function
    pixel_loss_fn = PixelLoss(criterion=cfg['pixel_criterion'])
    fea_loss_fn = ContentLoss(criterion=cfg['feature_criterion'])
    gen_loss_fn = GeneratorLoss(gan_type=cfg['gan_type'])
    edge_loss_fn = EdgeLoss(criterion=cfg['edge_criterion'])
    dis_loss_fn = DiscriminatorLoss(gan_type=cfg['gan_type'])

    # load checkpoint
    checkpoint_dir = './track2_unknown/checkpoints/' + cfg['sub_name']
    checkpoint = tf.train.Checkpoint(step=tf.Variable(0, name='step'),
                                     model=generator,
                                     discriminator=discriminator,
                                     optimizer_G=optimizer_G,
                                     optimizer_D=optimizer_D)

    manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                         directory=checkpoint_dir,
                                         max_to_keep=3)



    if not manager.latest_checkpoint:
        print("[*] training from scratch.")
        # loading pre-trained esrgan model for bootstraping
        if cfg['bootstrap_model'] is not None:
            # generate bootstrap model: esragan
            bootstrap_model = RRDB_Model(cfg['input_size'], cfg['ch_size'], cfg['network_G'])
            pretrain_dir = './track2_unknown/checkpoints/' + cfg['bootstrap_model']
            if tf.train.latest_checkpoint(pretrain_dir) and not manager.latest_checkpoint:
                checkpoint_pretrained = tf.train.Checkpoint(model=bootstrap_model,
                                                            )
                checkpoint_pretrained.restore(tf.train.latest_checkpoint(pretrain_dir))
                # extraction_variables = [var.value for var in generator.trainable_variables]
                # for i in range(len(extraction_variables)):
                #     generator.trainable_variables[i].value = extraction_variables[i]
                for var1,var2 in zip(generator.trainable_variables[:698],bootstrap_model.trainable_variables[:698]):
                    var1.assign(var2)
                del checkpoint_pretrained
                del pretrain_dir
                del bootstrap_model
                print('bootstrap from model [{}]'.format(cfg['bootstrap_model']))
        #del extraction_variables

    else:
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
        print('[*load ckpt from {} at step {}]'.format(manager.latest_checkpoint,checkpoint.step.numpy()))



    # define training step function
    @tf.function
    def train_step(lr, hr):
        with tf.GradientTape(persistent=True) as tape:
            sr = generator(lr, training=True)
            hr_output = discriminator(hr, training=True)
            sr_output = discriminator(sr, training=True)

            losses_G = {}
            losses_D = {}
            losses_G['reg'] = tf.reduce_sum(generator.losses)
            losses_D['reg'] = tf.reduce_sum(discriminator.losses)
            losses_G['pixel'] = cfg['w_pixel'] * pixel_loss_fn(hr, sr)
            losses_G['feature'] = cfg['w_feature'] * fea_loss_fn(hr, sr)
            losses_G['edge'] = cfg['w_edge'] * edge_loss_fn(hr,sr)
            losses_G['gan'] = cfg['w_gan'] * gen_loss_fn(hr_output, sr_output)
            losses_D['gan'] = dis_loss_fn(hr_output, sr_output)

            total_loss_G = tf.add_n([l for l in losses_G.values()])
            total_loss_D = tf.add_n([l for l in losses_D.values()])



        grads_G = tape.gradient(
            total_loss_G, generator.trainable_variables)

        grads_D = tape.gradient(
            total_loss_D, discriminator.trainable_variables)

        optimizer_G.apply_gradients(
            zip(grads_G, generator.trainable_variables))
        optimizer_D.apply_gradients(
            zip(grads_D, discriminator.trainable_variables))

        return total_loss_G, total_loss_D, losses_G, losses_D

    # training loop
    summary_writer = tf.summary.create_file_writer(
        './track2_unknown/logs/' + cfg['sub_name'])
    prog_bar = ProgressBar(cfg['niter'], checkpoint.step.numpy())
    remain_steps = max(cfg['niter'] - checkpoint.step.numpy(), 0)

    for lr, hr in train_dataset.take(remain_steps):
        checkpoint.step.assign_add(1)
        steps = checkpoint.step.numpy()

        total_loss_G, total_loss_D, losses_G, losses_D = train_step(lr, hr)

        prog_bar.update(
            "loss_G={:.4f}, loss_D={:.4f}, lr_G={:.1e}, lr_D={:.1e}".format(
                total_loss_G.numpy(), total_loss_D.numpy(),
                optimizer_G.lr(steps).numpy(), optimizer_D.lr(steps).numpy()))

        if steps % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    'loss_G/total_loss', total_loss_G, step=steps)
                tf.summary.scalar(
                    'loss_D/total_loss', total_loss_D, step=steps)
                for k, l in losses_G.items():
                    tf.summary.scalar('loss_G/{}'.format(k), l, step=steps)
                for k, l in losses_D.items():
                    tf.summary.scalar('loss_D/{}'.format(k), l, step=steps)

                tf.summary.scalar(
                    'learning_rate_G', optimizer_G.lr(steps), step=steps)
                tf.summary.scalar(
                    'learning_rate_D', optimizer_D.lr(steps), step=steps)

        if steps % cfg['save_steps'] == 0:
            manager.save()
            print("\n[*] save ckpt file at {}".format(
                manager.latest_checkpoint))

    print("\n [*] training done!")


def train(gpu,scale):
    # config files dictionary
    yaml_dicts = {2:'./track2_unknown/configs/esrgan_ea_x2_unknown.yaml',
                  3:'./track2_unknown/configs/esrgan_ea_x3_unknown.yaml',
                  4:'./track2_unknown/configs/esrgan_ea_x4_unknown.yaml'}

    if(scale in yaml_dicts):
        main(gpu,yaml_dicts[scale])
    else:
        print('Only support scale 2, scale 3 and scale 4.')

if __name__ == '__main__':
    #app.run(main)
    main()

