import os
import os.path as osp
import math
import time
import glob
import abc
from torch.utils.data import DataLoader
import torch.optim

from main.config import cfg


#
# def load_model(ckpt_path):
#     # model_file_list = glob.glob(osp.join(cfg.model_dir, '*.pth.tar'))
#     # cur_epoch = max(
#     #     [int(file_name[file_name.find('snapshot_') + 9: file_name.find('.pth.tar')]) for file_name in model_file_list])
#     # ckpt_path = osp.join(cfg.model_dir, 'snapshot_' + str(cur_epoch) + '.pth.tar')
#     ckpt = torch.load(ckpt_path)
#     start_epoch = ckpt['epoch'] + 1
#     # model.load_state_dict(ckpt['network'], strict=False)
#     # optimizer.load_state_dict(ckpt['optimizer'])
#
#     # info('Load checkpoint from {}'.format(ckpt_path))
#     return start_epoch
#
#
# load_model("/mnt/dms/KTW/hand4whole/weights/snapshot_6.pth.tar")

a = '-6.592985e-01	-8.463324e+00	5.806343e+00	-3.811371e-01	4.167802e+00	-1.099825e+01	-3.684745e+00	6.479623e+00	-5.881817e+00	-2.514634e-01	-3.423953e+00	-1.184184e+01	4.075499e+00	2.149731e+00	-6.690304e-01	1.084790e+00	-5.290133e-01	7.902027e-01	-7.068176e-01	9.766323e-01	-4.825064e-01	6.143519e-01	-3.246432e-01	3.808086e-01	-1.287797e+00	2.403593e+00	-6.216035e-01	8.999305e-01	-3.620210e-01	4.737992e-01	-2.743038e-01	3.733061e-01	-1.103727e+00	2.198267e+00	-6.374947e-01	8.806246e-01	-3.636793e-01	4.236057e-01	-2.053556e-01	3.186265e-01	-9.164085e-01	1.859152e+00	-5.929905e-01	7.818630e-01	-3.193474e-01	3.599475e-01	-1.755807e-01	2.645262e-01	-7.772086e-01	1.516473e+00	-4.372904e-01	5.708023e-01	-2.204752e-01	2.520307e-01	-1.532157e-01	2.149714e-01	5.317237e-01	6.864733e-01	5.922708e-01	4.891588e-01	8.256748e-01	6.929958e-01	6.099910e-01	4.368393e-01	4.562694e-01	2.987415e-01	1.501071e+00	1.786176e+00	6.892922e-01	6.305597e-01	4.132099e-01	3.023899e-01	3.290055e-01	2.593870e-01	1.269199e+00	1.660676e+00	7.066978e-01	5.653592e-01	4.146798e-01	2.921187e-01	2.702636e-01	2.802937e-01	1.016674e+00	1.283669e+00	6.499761e-01	4.948768e-01	3.643877e-01	2.657810e-01	2.438198e-01	2.559583e-01	7.971727e-01	8.875837e-01	4.746406e-01	3.735411e-01	2.447803e-01	1.752037e-01	1.907991e-01	1.969919e-01'
aa = a.split("	")
print(len(aa))