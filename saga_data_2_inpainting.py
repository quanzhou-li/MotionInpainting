import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True,
                        help='The path to the saga data')
    parser.add_argument('--save-name', required=True,
                        help='The location to save to results')
    args = parser.parse_args()
    data_path = args.data_path
    save_name = args.save_name

    ori_data = torch.load(data_path)
    motion_imgs = ori_data['I'][:,0,:330,:]
    res = {
        'motion_imgs': motion_imgs
    }
    torch.save(res, save_name)
