import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from acd import ACDModel, acd_model  # Assuming you have the ACDModel defined as above
import pdb



class StainNormalizer(object):
    def __init__(self, pixel_number=100000, step=300, batch_size=1500,
                 _template_dc_mat=None, _template_w_mat=None,
                 model_weights_path=None):
        self._pn = pixel_number
        self._bs = batch_size
        self._step_per_epoch = int(pixel_number / batch_size)
        self._epoch = int(step / self._step_per_epoch)
        self._template_dc_mat = _template_dc_mat
        self._template_w_mat = _template_w_mat
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ACDModel(input_dim=3) #.to(self.device)
        if model_weights_path:
            self.model.load_state_dict(torch.load(model_weights_path))
            self.model.eval()

        self.optimizer = optim.Adagrad(self.model.parameters(), lr=0.05)


    def fit(self, images):
        opt_cd_mat, opt_w_mat = self.extract_adaptive_cd_params(images)
        self._template_dc_mat = opt_cd_mat
        self._template_w_mat = opt_w_mat
        

    def transform(self, images):
        if self._template_dc_mat is None:
            raise AssertionError('Run fit function first')

        opt_cd_mat, opt_w_mat = self.extract_adaptive_cd_params(images)
        #print(opt_cd_mat,opt_w_mat)
        transform_mat = np.matmul(opt_cd_mat * opt_w_mat,
                                  np.linalg.inv(self._template_dc_mat * self._template_w_mat))

        od = -np.log((np.asarray(images, np.float32) + 1) / 256.0)
        normed_od = np.matmul(od, transform_mat)
        normed_images = np.exp(-normed_od) * 256 - 1

        return np.maximum(np.minimum(normed_images, 255), 0)

    def he_decomposition(self, images, od_output=True):
        if self._template_dc_mat is None:
            raise AssertionError('Run fit function first')

        opt_cd_mat, _ = self.extract_adaptive_cd_params(images)

        od = -np.log((np.asarray(images, np.float32) + 1) / 256.0)
        normed_od = np.matmul(od, opt_cd_mat)

        if od_output:
            return normed_od
        else:
            normed_images = np.exp(-normed_od) * 256 - 1
            return np.maximum(np.minimum(normed_images, 255), 0)

    def sampling_data(self, images):
        pixels = np.reshape(images, (-1, 3))
        pixels = pixels[np.random.choice(pixels.shape[0], min(self._pn * 20, pixels.shape[0]))]
        od = -np.log((np.asarray(pixels, np.float32) + 1) / 256.0)
        # filter the background pixels (white or black)
        tmp = np.mean(od, axis=1)
        #od = od[(tmp > 0.3) & (tmp < -np.log(30 / 256))]
        od = od[(tmp > 0.15) & (tmp < -np.log(30 / 256))]
        od = od[np.random.choice(od.shape[0], min(self._pn, od.shape[0]))]
        return od

    def extract_adaptive_cd_params(self, images):
        """
        :param images: RGB uint8 format in shape of [k, m, n, 3], where
                       k is the number of ROIs sampled from a WSI, [m, n] is 
                       the size of ROI.
        """
        od_data = self.sampling_data(images)
        #print(od_data.shape)
        input_od = torch.tensor(od_data, dtype=torch.float32) #.to(self.device)
        #print(input_od.shape)
        #input_od = torch.rand(None, 3, dtype=torch.float32)
        
        
        #input_od = tf.placeholder(dtype=tf.float32, shape=[None, 3])
        
        
        #target, cd, w = acd_model(input_od)
        #init = tf.global_variables_initializer()

        #model = ACDModel(input_od.shape[1])
        #model = self.model(input_od.shape[1])
        
        #optimizer = optim.Adagrad(model.parameters(), lr=0.05)
        #for param in model.parameters():
            #print(param)
        
        #target, cd, w = model(input_od)
        #print("*********************")
        #print(target, cd, w)
        
        #optimizer = optim.Adagrad(model.parameters(), lr=0.05)

        opt_cd=-1
        opt_w=-1
        for ep in range(self._epoch):
            for step in range(self._step_per_epoch):
                batch_od = input_od[step * self._bs:(step + 1) * self._bs]
                #print(batch_od.shape)
                self.optimizer.zero_grad()
                target, cd, w = self.model(batch_od)
                target.backward()
                self.optimizer.step()
                opt_cd=cd
                opt_w=w
        
        #with torch.no_grad():
        #    for param in 
        #    opt_cd = model.cd_mat.numpy()
        #    opt_w = [param.numpy() for param in model.w]
            
        opt_cd =  opt_cd.detach().numpy()
        opt_w = [param.detach().numpy() for param in opt_w]
        return opt_cd, opt_w
