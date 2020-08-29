import os
import cv2
import torchvision.transforms as transforms
import PIL
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image


## GradCAM
class GradCAM(object):
    def __init__(self, model, layers, img_path):
        self.model = model
        self.gradients = -1
        self.activations = -1
        self.layers = layers
        self.img_path = img_path
        self.img_list = list() 


        ## 定义前向传播的钩子
        ##Define hook for forward propagation
        def forward_hook(module, input, output):
            self.activations = output
            return None

        ## 定义反向传播的钩子
        ##Defining hooks for back propagation
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            return None 

        ## 钩子的实例化
        ##Instantiation of hook
        self.layers.register_forward_hook(forward_hook)
        self.layers.register_backward_hook(backward_hook)

    def ImagePreprocess(self):
        img = cv2.imread(self.img_path)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1]   # BGR --> RGB

        transform = transforms.Compose([   
        transforms.ToTensor(), 
        ])

        img = Image.fromarray(np.uint8(img))
        img_bf = transform(img)
        
        ## 把标准化前的img保存起来
        ##Keep the IMG before standardization
        nor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        img_af = nor(img_bf)

        # img_bf = img_bf.unsqueeze(0)

        # ## unsqueeze是增加一个维度，squeeze是去掉一个维度
        #Unsqueeze is to add a dimension, and squeeze is to remove a dimension
        img_af = img_af.unsqueeze(0)    # C*H*W --> B*C*H*W

        ## 返回标准化前的img和标准化后的img
        ##Return img before standardization and img after standardization
        return img_bf, img_af


    def forward(self, input):
        b, c, h, w = input.size()
        
        self.model.eval()

        prot = self.model(input)

        ## 反向传播 backward
        ##Backward propagation
        self.model.zero_grad()

        ## xx.max(1)会返回最大值和最大的索引
        
        ## To talk about xx.max (1) The maximum value and the largest index are returned

        ## prot[:, prot.max(1)[-1]]就是得到这个矩阵的某个索引
        ## prot[:, prot.max (1) [- 1]] is to get an index of the matrix
        
        ## 得到索引后是二维，然后.squeeze()用来降维
        ##After getting the index, it is two-dimensional, and then. Squeeze () is used to reduce the dimension
        # score = prot[:, prot.max(1)[-1]].squeeze()
        index = torch.argmax(prot)
        score = prot[:, index]

        score.backward(retain_graph=False)
        ## 得到activation和gradient以后就可以用来求取CAM图
        ##After the activation and gradient are obtained, they can be used to obtain cam drawings
        b, k, h, w = self.gradients.size()
        

        ## 将梯度的矩阵变成一行一行，然后在行方向上求取平均，相当于GAP,[1, 512]
        ##The gradient matrix is changed into a row by row, and then the average is obtained in the row direction, which is equivalent to gap, [1,512]
        alpha = self.gradients.view(b, k, -1).mean(2)

        ## 转化成[1, 512, 1, 1]的形式
        ##It is transformed into the form of [1, 512, 1, 1]
        weights = alpha.view(b, k, 1, 1)

        ## 将相乘得到的[1, 512, 14, 14]在第一维度上（512个）进行相加并且保留第一维度
        ##Add the multiplied [1, 512, 14, 14] on the first dimension (512) and keep the first dimension
        ## [1, 1, 14, 14]
        saliency_map = (weights*self.activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)

        ## 获取跟输入一样的大小
        ##Get the same size as the input
        _, _, h, w = input.size()

        ## 将特征图上采样到一样的大小
        ##Sample the feature map to the same size
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        ## 归一化
        ##Normalization
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data



        mask = saliency_map.cpu().data.numpy()

        return mask, index

    def HeatMap(self, mask, img):
        
        ## 制作heatmap，得到的通道数是在后面的
        ##When making Heatmap, the number of channels obtained is in the back
        heatmap = cv2.applyColorMap(np.uint8(255*mask.squeeze()), cv2.COLORMAP_JET)

        ## 转成torch的numpy并且调整一下通道数放前面
        ##Turn to torch's numpy and adjust the number of channels to the front
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)

        ## 在通道的维度上划分离出rgb三个矩阵，就是bgr转rgb
        ##In the channel dimension, the three separated RGB matrixes are BGR to RGB
        b, g, r = heatmap.split(1)

        heatmap = torch.cat([r, g, b])

        result = heatmap+img.cpu()

        result = result.div(result.max()).squeeze()

        return img, heatmap, result

    def __call__(self):

        ## 图像预处理
        ##Image preprocessing
        img_bf, img_af = self.ImagePreprocess()

        ## 前向传播
        ##Forward propagation and backward propagation
        mask, index = self.forward(img_af)

        ## 生成热度图
        ##Generate heat map
        img, heatmap, result = self.HeatMap(mask, img_bf)
        return img,heatmap,result,index



## GradCAM++  
class GradCAMPP(object):
    def __init__(self, model, layers, img_path):
        self.model = model
        self.gradients = -1
        self.activations = -1
        self.layers = layers
        self.img_path = img_path
        self.img_list = list() 


     
        def forward_hook(module, input, output):
            self.activations = output
            return None

        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
            return None 

      
        self.layers.register_forward_hook(forward_hook)
        self.layers.register_backward_hook(backward_hook)

    def ImagePreprocess(self):
        img = cv2.imread(self.img_path)
        img = cv2.resize(img, (224, 224))
        img = img[:, :, ::-1]   # BGR --> RGB

        transform = transforms.Compose([   
        transforms.ToTensor(), 
        ])

        img = Image.fromarray(np.uint8(img))
        img_bf = transform(img)
        
     
        nor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        img_af = nor(img_bf)

        # img_bf = img_bf.unsqueeze(0)

    
        img_af = img_af.unsqueeze(0)    # C*H*W --> B*C*H*W
      

      
        return img_bf, img_af


    def forward(self, input):
        b, c, h, w = input.size()
        
        self.model.eval()

        prot = self.model(input)

   
        self.model.zero_grad()


        # score = prot[:, prot.max(1)[-1]].squeeze()
        index = torch.argmax(prot)
        score = prot[:, index]

        score.backward(retain_graph=False)

        b, k, h, w = self.gradients.size()

        alpha_num = self.gradients.pow(2)

        alpha_denom = self.gradients.pow(2).mul(2) + self.activations.mul(self.gradients.pow(3)).view(b, k, h*w).sum(-1, keepdim=True).view(b, k, 1, 1)

        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

     
        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*self.gradients)

        weights = (alpha*positive_gradients).view(b, k, h*w).sum(-1).view(b, k, 1, 1)

     
        ## [1, 1, 14, 14]
        saliency_map = (weights*self.activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)

     
        _, _, h, w = input.size()

     
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)


        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data



        mask = saliency_map.cpu().data.numpy()

        return mask, index

    def HeatMap(self, mask, img):
        

        heatmap = cv2.applyColorMap(np.uint8(255*mask.squeeze()), cv2.COLORMAP_JET)

        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)


        b, g, r = heatmap.split(1)

        heatmap = torch.cat([r, g, b])

        result = heatmap+img.cpu()

        result = result.div(result.max()).squeeze()

        return img, heatmap, result

    def __call__(self):
        img_bf, img_af = self.ImagePreprocess()

        mask, index = self.forward(img_af)

        img, heatmap, result = self.HeatMap(mask, img_bf)

        return img, heatmap, result, index        
