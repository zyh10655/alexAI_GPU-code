import logging
import os
import torch
from os import path as osp
import sys
from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
# import hat_arch
# import hat_model
from basicsr.utils.options import dict2str, parse_options


# The root_path is to the config file of HAT model
def test_pipeline(root_path):
    try:
        def parse_options_wrapper(config_path):
            sys.argv = ['python', '-opt', config_path]  # Mock command-line arguments
            return parse_options(config_path)

        parse_options_wrapper(root_path)
        # parse options, set distributed setting, set random seed
        opt, _ = parse_options(root_path, is_train=False)
        root_dir = os.path.dirname(root_path)
        opt['path']['experiments_root'] = os.path.join(root_dir, "experiments/",
                                                       opt['name'])
        # Update the subdirectories
        opt['path']['models'] = os.path.join(opt['path']['experiments_root'], "models")
        opt['path']['training_states'] = os.path.join(opt['path']['experiments_root'], "training_states")
        opt['path']['log'] = os.path.join(opt['path']['experiments_root'])
        opt['path']['visualization'] = os.path.join(opt['path']['experiments_root'], "visualization")
        opt['path']['results_root'] = os.path.join(opt['path']['experiments_root'], "results_root")
        # Update the 'pretrain_network_g' path if necessary
        opt['path']['pretrain_network_g'] = os.path.join(root_dir,
                                                         "HAT-L_SRx2_ImageNet-pretrain.pth")

        # Update the 'pretrain_network_g' path if necessary
        # opt['path']['pretrain_network_g'] = os.path.join(root_dir, "/content/drive/MyDrive/HAT-L_SRx2_ImageNet-pretrain.pth")
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

        # mkdir and initialize loggers
        make_exp_dirs(opt)
        log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        # log_level = logging.INFO
        logger.info(get_env_info())
        logger.info(dict2str(opt))
        # Log the current GPU
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_gpu)
            logger.info(f'Using GPU for HAT: {current_gpu} - {gpu_name}')
        else:
            logger.info('No GPU available for HAT, using CPU.')
        # create test dataset and dataloader
        test_loaders = []
        for _, dataset_opt in sorted(opt['datasets'].items()):
            test_set = build_dataset(dataset_opt)
            #tile=opt['tile']['tile_size'],tile_pad=opt['tile']['tile_pad']
            test_loader = build_dataloader(
                test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
            logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
            test_loaders.append(test_loader)

        # create model
        model = build_model(opt)

        for test_loader in test_loaders:
            test_set_name = test_loader.dataset.opt['name']
            logger.info(f'Testing {test_set_name}...')
            model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    except Exception as e:
        print(f"TODO:Find the reason of this error, let's pick the image and move on for now")
        # print('All testing is done.')


