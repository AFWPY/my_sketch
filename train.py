from datasets import create_dataset
from model import create_model
import argparse
import time
import os

def setup_opt():
    """
    setup_opt
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
    parser.add_argument("--resume", default=None, help="path/to/saved/.pth")
    parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
    parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
    parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
    parser.add_argument('--n_epochs', type=int, default=10000, help='number of epochs with the initial learning rate')
    
    opt = parser.parse_args()

    return opt



def train(opt):
    # 创建数据集
    dataset = create_dataset(opt)  
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)

    #创建模型
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)

    # 确保这里的目录结构存在
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)

    # 文件路径
    log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

    # 这时候不需要检查文件是否存在，'a'模式会自动创建文件
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
    image_path = os.path.join(opt.checkpoints_dir, opt.name)

    for epoch in range(opt.n_epochs):
        epoch_start_time = time.time()  # timer for entire epoch
        # model.update_learning_rate()

        for i, data in enumerate(dataset):
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(i)

        model.save_images(image_path,epoch)

        # 保存模型
        if epoch % 10 == 0:
            model.save_networks('latest')
            model.save_networks(epoch)

        # 打印loss
        losses = model.get_current_losses()
        message = '(epoch: %d) ' % (epoch)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # print the message
        # 写入log
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs, time.time() - epoch_start_time))


def main():
    opt = setup_opt()
    train(opt)


if __name__ == "__main__":
    main()
