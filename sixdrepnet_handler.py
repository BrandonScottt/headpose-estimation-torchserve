import logging
import torch
import io
import os
import torch.backends.cudnn as cudnn
import time
import pickle
import utils
import numpy as np

from model import SixDRepNet
from PIL import Image
from torchvision import transforms as T
from ts.torch_handler.base_handler import BaseHandler

class SixDrepNetHandler(BaseHandler):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = T.Compose(
            [
                T.Resize(256),
                T.ToTensor(), 
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
            ]
        )
        self.model = None
        self.initialized = False
        self.device = None
    
    def initialize(self, context):
        cudnn.enabled = True
        properties = context.system_properties
        is_cuda = torch.cuda.is_available()
        gpu_id = properties.get("gpu_id")
        logging.info(F"CUDA is available: {is_cuda}")
        logging.info(F"GPU Device ID: {gpu_id}")

        # Only for Development
        self.map_location = "cuda" if is_cuda and gpu_id is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(gpu_id)
            if is_cuda and gpu_id is not None
            else self.map_location
        )

        model_dir = properties.get("model_dir")

        model_pth_path = os.path.join(model_dir, '6DRepNet_300W_LP_BIWI.pth')
        
        model_def_path = os.path.join(model_dir, "model.py")
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing model definition file")

        state_dict = torch.load(model_pth_path, map_location=self.device)
        self.model = SixDRepNet(backbone_name='RepVGG-B1g2',
                        backbone_file='',
                        deploy=True,
                        pretrained=False)

        if 'model_state_dict' in state_dict:
            self.model.load_state_dict(state_dict['model_state_dict'])
        else:
            self.model.load_state_dict(state_dict) 
        self.model.eval()
        self.model.to(self.device)

        self.initialized = True
    
    def pickle_to_form(self, hits):
       
        converted_hits = []
        conversion_times = []
        for pickled_list_dict in hits:
            start_time = time.perf_counter()
            assert len(pickled_list_dict) == 1, "This handler only accepts 1 pickled list per request (hit)."
            pickled_list = list(pickled_list_dict.values())[0]
            converted_hit = pickle.loads(pickled_list)
            converted_hits.append(dict(enumerate(converted_hit)))
            conversion_times.append(time.perf_counter() - start_time)
        return converted_hits, conversion_times

    def preprocess(self, hit):
        transform_images = []

        for image_bytes, bbox in hit.values():
            
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            #crop image using bounding box from detection
            area = (bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3])
            image = image.crop(area)
            image = self.transform(image)

            transform_images.append(image)

        return transform_images

    def inference(self, hit):
        result = []

        for image in hit:

            self.model.eval()
            image = np.expand_dims(image, 0)
            image = torch.Tensor(image).cuda()

            outputs = self.model(image)

            euler = utils.compute_euler_angles_from_rotation_matrices(outputs)*180/np.pi

            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            yaw=y_pred_deg[0].tolist()
            pitch=p_pred_deg[0].tolist()
            roll=r_pred_deg[0].tolist()

            result.append({
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            })

        return result

    def postprocess(self, hit):
        return hit
    
    def multihit_preprocess(self, hits):

        # convert
        hits, conversion_times = self.pickle_to_form(hits)

        # normally preprocess
        preprocessed_hits = []
        preprocess_times = []
        for hit in hits:
            start_time = time.perf_counter()
            preprocessed_hits.append(self.preprocess(hit))
            preprocess_times.append(time.perf_counter() - start_time)

        # add conversion times with preprocess times
        total_times = [c + p for c, p in zip(conversion_times, preprocess_times)]

        return preprocessed_hits, total_times

    def multihit_inference(self, hits):
        inferenced_hits = []
        inference_times = []
        for hit in hits:
            start_time = time.perf_counter()
            inferenced_hits.append(self.inference(hit))
            inference_times.append(time.perf_counter() - start_time)
        return inferenced_hits, inference_times

    def multihit_postprocess(self, hits):
        postprocessed_hits = []
        postprocess_times = []
        for hit in hits:
            start_time = time.perf_counter()
            postprocessed_hits.append(self.postprocess(hit))
            postprocess_times.append(time.perf_counter() - start_time)

        return postprocessed_hits, postprocess_times
    
    def construct_output(
        self, 
        hits,preprocess_times,
        inference_times,
        postprocess_times
    ):
        outputs = []
        for hit, prep, infer, post in zip(
            hits, 
            preprocess_times, 
            inference_times, 
            postprocess_times
        ):
            output = {
                "result": hit,
                "preprocess_time": prep,
                "inference_time": infer,
                "postprocess_time": post,
            }
            outputs.append(output)
        return outputs

    def handle(self, data, context):
        if not self.initialized:
            self.initialize(context)
            
        if data is None:
            return None

        hits = data
        hits, preprocess_times = self.multihit_preprocess(hits)
        hits, inference_times = self.multihit_inference(hits)
        hits, postprocess_times = self.multihit_postprocess(hits)

        return self.construct_output(
            hits,
            preprocess_times,
            inference_times,
            postprocess_times
        )