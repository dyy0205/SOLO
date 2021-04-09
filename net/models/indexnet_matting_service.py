# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
try:
    from net.matting_networks.transforms import trimap_transform, groupnorm_normalise_image
    from net.matting_networks.models import build_model
except ModuleNotFoundError:
    from matting_networks.transforms import trimap_transform, groupnorm_normalise_image
    from matting_networks.models import build_model
import cv2
import logging

# ignore warnings
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger('django')

FBA_MODEL_PATH = './ckpt/FBA.pth'

# 根据 mask 生成 Trimap
def gen_tri(fg_mask, kernel_size=20):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

    fg = fg_mask.astype(np.float)
    fg_max = fg.flatten().max()
    fg_min = fg.flatten().min()

    fg = ((fg - fg_min) / (fg_max - fg_min) * 255).astype(np.uint8)

    fg_mask = np.zeros(fg.shape, dtype=np.uint8)
    bg_mask = np.zeros(fg.shape, dtype=np.uint8)

    fg_mask[fg >= 250] = 1
    bg_mask[fg == 0] = 1
    fg_mask = cv2.erode(fg_mask, kernel)
    bg_mask = cv2.erode(bg_mask, kernel)

    trimap = np.zeros(fg.shape, dtype=np.uint8)
    trimap[fg_mask == 1] = 255
    trimap[(trimap == 0) & (bg_mask == 0)] = 128

    return trimap

class Matting_Model():
    def __init__(self, model_path):
        logger.info('&&&&&&&&&&&&&&&&&&& Matting model loading &&&&&&&&&&&&&&&')
        self.net = build_model()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        sd = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(sd, strict=True)
        self.net.eval()
    
    def np_to_torch(self, x):
        return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cuda()
        # return torch.from_numpy(x).permute(2, 0, 1)[None, :, :, :].float().cpu()

    def scale_input(self, x: np.ndarray, scale: float, scale_type) -> np.ndarray:
        ''' Scales inputs to multiple of 8. '''
        h, w = x.shape[:2]
        h1 = int(np.ceil(scale * h / 8) * 8)
        w1 = int(np.ceil(scale * w / 8) * 8)
        x_scale = cv2.resize(x, (w1, h1), interpolation=scale_type)
        return x_scale

    def convert_trimap(self, trimap_im):
        # trimap_im = trimap_im / 255.
        h, w = trimap_im.shape
        trimap = np.zeros((h, w, 2))
        trimap[trimap_im == 255, 1] = 1
        trimap[trimap_im == 0, 0] = 1
        return trimap

model = Matting_Model(FBA_MODEL_PATH)

def inference(image_cv, mask=None, trimap=None, edge_optimize=False):
    '''
    image_cv: h,w,c 0-255
    trimap: h,w 0,128,255
    '''
    with torch.no_grad():
        if trimap is None:
            assert image_cv.shape[:2] == mask.shape[:2], "image's size({}) if different from mask's size{} ".format(
                image_cv.shape, mask.shape
            )
            trimap = gen_tri(mask, kernel_size=15)

        image_np = (image_cv / 255.0)[:, :, ::-1]
        trimap_np = model.convert_trimap(trimap)
        h, w = trimap_np.shape[:2]

        image_scale_np = model.scale_input(image_np, 1.0, cv2.INTER_LANCZOS4)
        trimap_scale_np = model.scale_input(trimap_np, 1.0, cv2.INTER_LANCZOS4)

        image_torch = model.np_to_torch(image_scale_np)
        trimap_torch = model.np_to_torch(trimap_scale_np)

        trimap_transformed_torch = model.np_to_torch(trimap_transform(trimap_scale_np))
        image_transformed_torch = groupnorm_normalise_image(image_torch.clone(), format='nchw')

        output = model.net(image_torch, trimap_torch, image_transformed_torch, trimap_transformed_torch)
        output = cv2.resize(output[0].cpu().numpy().transpose((1, 2, 0)), (w, h), cv2.INTER_LANCZOS4)

        alpha = output[:, :, 0]
        fg = output[:, :, 1:4]
        bg = output[:, :, 4:7]

        alpha[trimap_np[:, :, 0] == 1] = 0
        alpha[trimap_np[:, :, 1] == 1] = 1
        fg[alpha == 1] = image_np[alpha == 1]
        bg[alpha == 0] = image_np[alpha == 0]

        #############
        ### 去黑边 ###
        #############
        if edge_optimize:
            image_np = (image_cv / 255.0)
            base = fg.copy()[:, :, ::-1]
            base1 = np.multiply(base,alpha[...,np.newaxis])
            base = image_np.copy()
            base2 = np.multiply(base,alpha[...,np.newaxis])
            diff = np.abs(base2-base1)
            if diff.max() > 0.10:
                diff = (diff*255).astype(np.uint8)

                diff = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)

                diff = diff/255.

                copy = diff.copy()
                thresh1 = np.max(copy)/10
                thresh2 = np.max(copy)/20
                diff[copy>thresh1] = 5*copy[copy>thresh1]
                diff[copy<thresh2] = 0#0.5*copy[copy<thresh2]
                diff = np.clip(diff,0,1)

                canny = cv2.Canny((diff*255).astype(np.uint8),250,254,9)
                canny = cv2.blur(canny,(5,5))
                canny = cv2.blur(canny,(4,4))
                canny = np.clip((3*(canny/255.)),0,1)
                canny = cv2.erode(canny, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=2)

                undesired = cv2.blur(canny*255,(30,1))
                undesired = cv2.blur(undesired,(1,30))
                thresh = 6/30
                undesired = ((undesired/255.)>thresh).astype(np.float32)
                undesired = cv2.dilate(undesired, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)#

                diff = diff*canny*(1-undesired)
                copy = diff.copy()
                thresh1 = np.max(copy)/3
                thresh2 = np.max(copy)/10
                diff[copy>thresh1] = (copy[copy>thresh1]/thresh1)*copy[copy>thresh1]
                diff[copy<thresh2] = 0#0.5*copy[copy<thresh2]
                diff = np.clip(diff,0,1)

                alpha_noH = alpha.copy()
                alpha_noH[trimap_np[:, :, 0] == 1] = 0
                alpha_noH[trimap_np[:, :, 1] == 1] = 1
                alpha_noH = cv2.erode(alpha_noH, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=8)
                alpha_noH = cv2.dilate(alpha_noH, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)),iterations=8)
                alpha_noH = cv2.blur(alpha_noH,(5,5))
                alpha_noH = (alpha_noH>0.75).astype(np.float32)
                alpha_noH = cv2.dilate(alpha_noH, cv2.getStructuringElement(cv2.MORPH_RECT,(4,4)),iterations=1)#
                # alpha_noH = cv2.dilate(alpha_noH, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=2)  #

                diff = diff * alpha_noH

                diff_big = cv2.dilate(diff, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=4)#
                diff_big = diff_big - (alpha>0.5).astype(np.float32)
                diff_big = np.clip(diff_big,0,1)
                diff = np.clip((diff+diff_big),0,1)

                res = np.clip((alpha-diff),0,1)
                res[trimap_np[:, :, 0] == 1] = 0
                res[trimap_np[:, :, 1] == 1] = 1
                alpha = res
        ### 去黑边结束 ###

        alpha = (alpha * 255).astype(np.uint8)
        fg = (fg * 255).astype(np.uint8)
        return alpha, fg


def model_matting(**kwargs):
    model = Matting_Model(FBA_MODEL_PATH)
    return model


def convert_model():
    model = Matting_Model(FBA_MODEL_PATH)
    x = torch.rand(1, 3, 576, 576, device=device)
    trimap = torch.rand(1, 2, 576, 576, device=device)
    image_trans = torch.rand(1, 3, 576, 576, device=device)
    trimap_trans = torch.rand(1, 6, 576, 576, device=device)

    output = model.net(x, trimap, image_trans, trimap_trans)
    # =========================================================================
    # STEP video. import converter helper and hood functional
    #   If you define the network in other file, you need to make sure the
    #   functional is imported from converter_helper
    from converter_helper import functional as F
    import converter_helper

    converter_helper.set_convert_mode(True)
    # =========================================================================
    # STEP 2. PyTorch -> ONNX
    #    Then the onnx model is in a local file called result.onnx
    model = converter_helper.pytorch_to_onnx(model.net, (x, trimap, image_trans, trimap_trans),
                                             input_names=['image', 'trimap', 'image_trans', 'trimap_trans'],
                                             output_names=['alpha'])
    # =========================================================================
    # STEP 3. ONNX -> CoreML
    coreml_model = converter_helper.onnx_to_coreml(
        model,
        image_input_names=['image'],  # a list of input names to be used as image
        image_output_names=['alpha'],  # a list of output names to be used as image
    )
    # add some fid info if you need
    fid = 'EDITING_HUM_SEG12'
    model_name = 'FBA'
    coreml_model = converter_helper.coreml_add_fid(coreml_model, model_name, fid)
    # save the coreml to local file if you need
    coreml_model.save('model.mlmodel')
    # =========================================================================
    # STEP 4. CoreML -> SDK
    #   If you dont need to convert the model to encryption sdk model,
    #   you should omit this part.
    filename = fid + '_' + model_name
    converter_helper.convert_model_to_sdk(coreml_model, filename)
convert_model()
exit()

if __name__ == '__main__':
    import os
    image_path = "./test_video/video"
    save_root = "./test_video/matting"
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for r, dirs, files in os.walk(image_path):
        for f in files:
            if f.split(".")[-1] == "jpg":
                path = os.path.join(image_path, f)
                x = cv2.imread(path)
                label_path = f.split(".")[0] +".png"
                m = cv2.imread(os.path.join(image_path, label_path), flags=0)
    # x = cv2.imread("./testdata/test_picture/pic/100.jpg")
    # m = cv2.imread("./testdata/test_picture/label/100.png", flags=0)

    # alpha = inference(x, m).to(device)
                alpha = inference(x, m)
                save_path = os.path.join(save_root, label_path)
                cv2.imwrite(save_path, alpha)
    print("END")


