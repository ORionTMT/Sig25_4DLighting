import glob
import json
import os
import skimage
import numpy as np
from pathlib import Path
from natsort import natsorted
from PIL import Image
from .image_processor import pil_square_image, center_crop
from .camera_utils import get_intrinsic_matrix, get_projection_matrix, focal2fov, fov2focal
from tqdm.auto import tqdm
import random
import itertools
from abc import ABC, abstractmethod
import cv2
import torch

class Dataset(ABC):
    def __init__(self,
                 resolution=(1024, 1024),
                 force_square=True,
                 return_image_path=False,
                 return_dict=False,
        ):
        """
        Resoution is (WIDTH, HEIGHT)
        """
        self.resolution = resolution
        self.force_square = force_square
        self.return_image_path = return_image_path
        self.return_dict = return_dict
        self.scene_data = []
        self.meta_data = []
        self.boundary_info = []
        
    @abstractmethod
    def _load_data_path(self):
        pass

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        image = Image.open(self.scene_data[idx])
        if self.force_square:
            # image = pil_square_image(image, self.resolution)
            image = center_crop(image).resize(self.resolution)
        else:
            image = image.resize(self.resolution)
        
        if self.return_dict:
            d = {
                "image": image,
                "path": self.scene_data[idx]
            }
            if len(self.boundary_info) > 0:
                d["boundary"] = self.boundary_info[idx]
                
            return d
        elif self.return_image_path:
            return image, self.scene_data[idx]
        else:
            return image

class GeneralLoader(Dataset):
    def __init__(self,
                 root=None,
                 num_samples=None,
                 res_threshold=((1024, 1024)),
                 apply_threshold=False,
                 random_shuffle=False,
                 process_id = 0,
                 process_total = 1,
                 limit_input = 0,
                 **kwargs,
        ):
        super().__init__(**kwargs)
        self.root = root if os.path.isdir(root) else os.path.dirname(root)
        self.res_threshold = res_threshold
        self.apply_threshold = apply_threshold
        self.has_meta = False
        
        if self.root is not None:
            if not os.path.exists(self.root):
                raise Exception(f"Dataset {self.root} does not exist.") 
            
            if os.path.isdir(root):
                paths = natsorted(
                    list(glob.glob(os.path.join(self.root, "*.png"))) + \
                    list(glob.glob(os.path.join(self.root, "*.jpg")))
                )
            else:
                paths = [root]
            self.scene_data = self._load_data_path(paths, num_samples=num_samples)
            
            if random_shuffle:
                SEED = 0
                random.Random(SEED).shuffle(self.scene_data)
                random.Random(SEED).shuffle(self.boundary_info)
            
            if limit_input > 0:
                self.scene_data = self.scene_data[:limit_input]
                self.boundary_info = self.boundary_info[:limit_input]
                
            # please keep this one the last, so, we will filter out scene_data and boundary info
            if process_total > 1:
                self.scene_data = self.scene_data[process_id::process_total]
                self.boundary_info = self.boundary_info[process_id::process_total]
                print(f"Process {process_id} has {len(self.scene_data)} samples")

    def _load_data_path(self, paths, num_samples=None):
        if os.path.exists(os.path.splitext(paths[0])[0] + ".json") or os.path.exists(os.path.splitext(paths[-1])[0] + ".json"):
            self.has_meta = True
        
        if self.has_meta:
            # read metadata
            TARGET_KEY = "chrome_mask256"
            for path in paths:
                with open(os.path.splitext(path)[0] + ".json") as f:
                    meta = json.load(f)
                    self.meta_data.append(meta)
                    boundary =  {
                        "x": meta[TARGET_KEY]["x"],
                        "y": meta[TARGET_KEY]["y"],
                        "size": meta[TARGET_KEY]["w"],
                    }
                    self.boundary_info.append(boundary)
                
        
        scene_data = paths
        if self.apply_threshold:
            scene_data = []
            for path in tqdm(paths):
                img = Image.open(path)
                if (img.size[0] >= self.res_threshold[0]) and (img.size[1] >= self.res_threshold[1]):
                    scene_data.append(path)
        
        if num_samples is not None:
            max_idx = min(num_samples, len(scene_data))
            scene_data = scene_data[:max_idx]
        
        return scene_data
    
    @classmethod
    def from_image_paths(cls, paths, *args, **kwargs):
        dataset = cls(*args, **kwargs)
        dataset.scene_data = dataset._load_data_path(paths)
        return dataset

class LocalImagesLoader(Dataset):
    def __init__(self, root_dir=None, resolution=(1024, 1024), **kwargs):
        super().__init__(**kwargs)
        assert root_dir is not None, "root_dir must be provided."
        assert os.path.isdir(root_dir), f"{root_dir} is not a valid directory."
        
        self.root_dir = root_dir
        self.resolution = resolution  # Target resolution for resizing images

        # Load all images with common extensions (png, jpg, jpeg)
        image_paths = sorted(glob.glob(os.path.join(root_dir, "*.*")))
        valid_extensions = ['.png', '.jpg', '.jpeg']
        self.image_paths = [path for path in image_paths if os.path.splitext(path)[1].lower() in valid_extensions]

        self.scene_data = []
        for img_path in self.image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                
                # Crop the image to its largest centered square
                width, height = image.size
                min_side = min(width, height)
                left = (width - min_side) // 2
                top = (height - min_side) // 2
                right = left + min_side
                bottom = top + min_side
                image = image.crop((left, top, right, bottom))  # Crop the image
                
                # Resize the cropped image to the target resolution
                image = image.resize(self.resolution)  
                
                self.scene_data.append({
                    "image": image,
                    "path": img_path,
                })
            except Exception as e:
                print(f"Error loading image from {img_path}: {e}")

    def __len__(self):
        return len(self.scene_data)

    def _load_data_path(self, paths, num_samples=None):
        # No implementation needed for now, or you can add logic here later
        pass

    def __getitem__(self, idx):
        d = self.scene_data[idx]
        return d

class SimpleImageLoader(Dataset):
    def __init__(self,
                 image_path=None,
                 resolution=(512, 512),
                 **kwargs):
        super().__init__(**kwargs)

        assert os.path.isfile(image_path), "提供的路径不是一个有效的图片文件"
        self.image_path = image_path
        self.resolution = resolution
        
        # 加载图片
        image = Image.open(image_path).convert("RGB")
        image = center_crop(image).resize(self.resolution)  # 调整图片到指定分辨率
        
        # 将图片保存到字典中
        self.scene_data = [{"image": image, "path": image_path}]
        
    def __len__(self):
        return len(self.scene_data)
    def _load_data_path(self, paths, num_samples=None):
        # No implementation needed for now, or you can add logic here later
        pass
    def __getitem__(self, idx):
        # 返回包含图片的字典
        return self.scene_data[idx]

import os
import cv2
from PIL import Image
from torch.utils.data import Dataset

class VideoFrameLoader(Dataset):
    def __init__(self,
                 video_path=None,
                 resolution=(512, 512),
                 read_on_init=False,  # 是否在初始化时把全部帧读入内存
                 **kwargs):
        super().__init__(**kwargs)
        print("video_path:", video_path)
        assert isinstance(video_path, str) and os.path.isfile(video_path), "提供的路径不是一个有效的视频文件"
        self.video_path = video_path
        self.resolution = resolution
        self.read_on_init = read_on_init

        self.video = cv2.VideoCapture(video_path)
        assert self.video.isOpened(), "无法打开视频文件"

        # 尽量拿到帧数，但以实际读取为准（有些编码器帧数不准确）
        self.total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
        print("reported total_frames:", self.total_frames)

        self._frames_in_memory = None
        if self.read_on_init:
            self._frames_in_memory = self._read_all_frames(self.video)
            # 读完就释放
            self.video.release()

    def __len__(self):
        # 数据集只包含一个视频
        return 1

    def __getitem__(self, idx):
        """
        返回该视频的所有帧（PIL 图像列表）
        """
        if self._frames_in_memory is not None:
            frames = self._frames_in_memory
        else:
            # 每次从头读完整视频
            if not self.video.isOpened():
                self.video.open(self.video_path)
            # 回到起始位置
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frames = self._read_all_frames(self.video)
        return {"frames": frames}

    def _read_all_frames(self, cap):
        """
        从当前位置开始，依次读取直到结束。
        始终以实际读取的帧为准。
        """
        frames = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.resolution is not None:
                frame = cv2.resize(frame, self.resolution)
            frames.append(Image.fromarray(frame))
        if self.total_frames != -1 and len(frames) != self.total_frames:
            print(f"[WARN] 实际读取帧数 {len(frames)} 与报告帧数 {self.total_frames} 不一致（以实际为准）")
        return frames

    def __del__(self):
        if hasattr(self, 'video') and self.video is not None and self.video.isOpened():
            self.video.release()

    def _load_data_path(self, paths, num_samples=None):
        pass

class PosedImagesLoader(Dataset):
    def __init__(self,
                 root=None,
                 num_samples=4,
                 factor=4,
                 load_ref=True,
                 **kwargs,
        ):
        super().__init__(**kwargs)
        print("root", root)
        assert os.path.isdir(root)
        self.root = root

        meta_path = os.path.join(root, "transforms.json")
        with open(meta_path) as f:
            meta_data = json.load(f)

        if 'fl_x' in meta_data:
            assert 'w' in meta_data
            fov_x = focal2fov(meta_data['fl_x'], meta_data['w'])
        else:
            assert 'camera_angle_x' in meta_data
            fov_x = meta_data['camera_angle_x']

        n_total_frames = len(meta_data["frames"])
        subset_indices = np.linspace(0, n_total_frames-1, num_samples, dtype=int).tolist()

        ref_image_dir = os.path.join(root, "render_persp")
        ref_image_paths = None
        if load_ref and os.path.exists(ref_image_dir):
            ref_image_paths = sorted(glob.glob(os.path.join(ref_image_dir, "*.png")))

        self.scene_data = []
        for i in subset_indices:
            frame_meta = meta_data["frames"][i]
            img_path = frame_meta["file_path"]
            img_path = os.path.join(root, img_path)
            if factor > 1:
                img_factor_path = img_path.replace("images", f"images_{factor}")
                if os.path.exists(img_factor_path):
                    img_path = img_factor_path
            pose = np.array(frame_meta["transform_matrix"])

            image = Image.open(img_path)
            W, H = image.size
            fov_y = focal2fov(fov2focal(fov_x, W), H)
            print("W, H", W, H)
            print("fovy", fov_y)
            print("fovx", fov_x)
            image = center_crop(image).resize(self.resolution)
            # image = pil_square_image(image, self.resolution)
            W, H = image.size
            if fov_x < fov_y:
                focal = fov2focal(fov_x, W)
            else:
                focal = fov2focal(fov_y, H)
            K = get_intrinsic_matrix(focal, W, H)
            P = get_projection_matrix(K, pose)

            frame_dict = {
                "image": image,
                "path": img_path,
                "pose": pose,
                "K": K,
                "projection": P,
            }

            if load_ref and ref_image_paths:
                ref_image_path = ref_image_paths[i]
                ref_image = Image.open(ref_image_path).convert("RGB")
                ref_image = center_crop(ref_image).resize(self.resolution)
                frame_dict["ref_image"] = ref_image

            self.scene_data.append(frame_dict)


    def _load_data_path(self, paths, num_samples=None):
        pass

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        if self.return_dict:
            d = self.scene_data[idx]
            return d
        elif self.return_image_path:
            image = self.scene_data[idx]["image"]
            pose = self.scene_data[idx]["pose"]
            return image, pose
        else:
            image = self.scene_data[idx]["image"]
            return image


def read_depth(path):
    if path.endswith(".png"):
        depth = cv2.imread(path)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
        depth = depth.astype(np.float32) / 255.0
    elif path.endswith(".exr"):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = 1.0 / (depth.astype(np.float32) + 1e-6)  # convert to disparity
        depth_min = np.min(depth)
        depth_max = np.max(depth)
        depth = (depth - depth_min) / (depth_max - depth_min)
        depth = np.clip(depth, 0, 1)
        depth_8bit = (depth * 255).astype(np.uint8)

        # 使用 PIL 将 NumPy 数组转换为 Image 格式
        depth_pil = Image.fromarray(depth_8bit)
        depth_pil = depth_pil.resize((1024, 1024))

        
    else:
        raise ValueError(f"Unsupported depth format: {path}")
    depth_resized = cv2.resize(depth, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    return depth_pil


def read_image_exr(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = np.where(image == np.inf, 65536, image)
    image = (image * 255.0).astype(np.uint8)  # Convert to uint8 for PIL

    return Image.fromarray(image).resize((1024, 1024))
    

def read_mask(path):
    mask = Image.open(path).convert("L")
    mask = np.asarray(mask).astype(np.float32)[..., None] / 255.0
    return mask



class CustomEvalDataloader(Dataset):
    def __init__(self, test_samples, base_eval_data_dir, val_set, json_path, saving_json_path=None, use_wrapped_cond=False):
        self.base_eval_data_dir = base_eval_data_dir
        self.val_set = val_set
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.eval_data = json.load(f)
        if test_samples < len(self.eval_data):
            self.eval_data = random.sample(self.eval_data, test_samples)

        self.test_samples = min(test_samples, len(self.eval_data))
        self.use_wrapped_cond = use_wrapped_cond
        #save the eval_data to a json file
        # 检查是否已经存在保存的 JSON 文件，如果存在，读取数据
        '''
        if os.path.exists(saving_json_path):
            with open(saving_json_path, 'r') as f:
                saved_eval_data = json.load(f)
        #检查根目录是否存在
        else:
            if not os.path.exists(os.path.dirname(saving_json_path)):
                os.makedirs(os.path.dirname(saving_json_path))
            saved_eval_data = []


        # 合并新的 eval_data 和已有的数据
        saved_eval_data.extend(self.eval_data)

        # 保存合并后的数据，并确保格式化
        with open(saving_json_path, 'w') as f:
            json.dump(saved_eval_data, f, indent=4)
        '''

    def __len__(self):
        return self.test_samples
    def _load_data_path(self, paths, num_samples=None):
        pass

    def __getitem__(self, idx):
        item = self.eval_data[idx]

        if self.val_set == "3dfront":
            bg_path = os.path.join(self.base_eval_data_dir, item.get('background_rgb'))
            bg_path = bg_path[:-4] + ".png"
            mask_path = os.path.join(self.base_eval_data_dir, item.get('sphere_mask'))
            depth_path = os.path.join(self.base_eval_data_dir, item.get('sphere_depth'))
            gt_path = os.path.join(self.base_eval_data_dir, item.get('sphere_rgb'))
            image_name = item.get("room_id") + str(item.get("sphere_id")) + str(item.get("view_id"))
            if self.use_wrapped_cond:
                sphere_path = os.path.join(self.base_eval_data_dir, item.get("wrapped_sphere_path"))
        elif item["data_type"] == "extra":
            bg_path = item.get('background_rgb')
            bg_path = bg_path[:-4] + ".png"
            mask_path = item.get('sphere_mask')
            depth_path = item.get('sphere_depth')
            gt_path = item.get('sphere_rgb')

        elif self.val_set == "infinigen":
            bg_path = item.get('background_rgb')
            bg_path = bg_path[:-4] + ".png"
            mask_path = item.get('sphere_mask')
            depth_path = item.get('sphere_depth')
            gt_path = item.get('sphere_rgb')
            pose = item.get('camera_pose')
            fov = item.get('camera_angle_x')
            focal = fov2focal(fov, 1024)
            K = get_intrinsic_matrix(focal, 1024, 1024)
            image_name = item.get("room_id") + str(idx)
            if self.use_wrapped_cond:
                sphere_path = item.get("wrapped_sphere_path")
        else:
            raise ValueError("Invalid val_set. Must be either '3dfront' or 'infinigen'.")

        example = {}


        image = Image.open(bg_path).convert("RGB")
        image = image.resize((1024, 1024))
        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((1024, 1024))
        #change the gt path from .exr to .png
        gt_path = gt_path[:-4] + ".png"
        groundtruth = Image.open(gt_path).convert("RGB")
        groundtruth = groundtruth.resize((1024, 1024))
        if self.val_set == "infinigen":
            if not item["data_type"] == "extra":
                depth= read_depth(depth_path)
                depth = [depth]
        if self.val_set == "3dfront":
            depth= read_depth(depth_path)
            depth = [depth]
        mask = [mask]


        # Convert depth to a PIL image as well (if needed)
        if self.val_set == "infinigen":
            if item["data_type"] == "extra":
                depth = Image.open(depth_path).convert("L")
                depth = depth.resize((1024, 1024))
                example["image"] = image
                example["mask"] = mask
                example["depth"] = depth
                example["path"] = bg_path
                groundtruth = Image.open(gt_path).convert("RGB")
                groundtruth = groundtruth.resize((1024, 1024))
                example["gt"] = groundtruth
            else:
                example["image"] = image
                example["pose"] = pose
                example["K"] = K
                example["mask"] = mask
                example["depth"] = depth
                example["path"] = image_name
                example["gt"] = groundtruth
                example["pixel_values_path"] = gt_path
                example["masked_images_path"] = bg_path
                example["masks_path"] = mask_path
                example["depth_values_path"] = depth_path
        else:
                example["image"] = image

                example["mask"] = mask
                example["depth"] = depth
                example["path"] = image_name
                example["gt"] = groundtruth
                example["pixel_values_path"] = gt_path
                example["masked_images_path"] = bg_path
                example["masks_path"] = mask_path
                example["depth_values_path"] = depth_path

        return example
    
class EvalDataloader(Dataset):
    def __init__(self, test_samples, base_eval_data_dir, val_set, json_path, saving_json_path, use_wrapped_cond=False):
        self.base_eval_data_dir = base_eval_data_dir
        self.val_set = val_set
        self.json_path = json_path
        with open(json_path, 'r') as f:
            self.eval_data = json.load(f)
        if test_samples < len(self.eval_data):
            self.eval_data = random.sample(self.eval_data, test_samples)

        self.test_samples = min(test_samples, len(self.eval_data))
        self.use_wrapped_cond = use_wrapped_cond
        #save the eval_data to a json file
        # 检查是否已经存在保存的 JSON 文件，如果存在，读取数据
        '''
        if os.path.exists(saving_json_path):
            with open(saving_json_path, 'r') as f:
                saved_eval_data = json.load(f)
        #检查根目录是否存在
        else:
            if not os.path.exists(os.path.dirname(saving_json_path)):
                os.makedirs(os.path.dirname(saving_json_path))
            saved_eval_data = []


        # 合并新的 eval_data 和已有的数据
        saved_eval_data.extend(self.eval_data)

        # 保存合并后的数据，并确保格式化
        with open(saving_json_path, 'w') as f:
            json.dump(saved_eval_data, f, indent=4)
        '''

    def __len__(self):
        return self.test_samples
    def _load_data_path(self, paths, num_samples=None):
        pass

    def __getitem__(self, idx):
        item = self.eval_data[idx]

        if self.val_set == "3dfront":
            bg_path = os.path.join(self.base_eval_data_dir, item.get('background_rgb'))
            bg_path = bg_path[:-4] + ".png"
            mask_path = os.path.join(self.base_eval_data_dir, item.get('sphere_mask'))
            depth_path = os.path.join(self.base_eval_data_dir, item.get('sphere_depth'))
            gt_path = os.path.join(self.base_eval_data_dir, item.get('sphere_rgb'))
            image_name = item.get("room_id") + str(item.get("sphere_id")) + str(item.get("view_id"))
            if self.use_wrapped_cond:
                sphere_path = os.path.join(self.base_eval_data_dir, item.get("wrapped_sphere_path"))
        elif self.val_set == "infinigen":
            bg_path = item.get('background_rgb')
            bg_path = bg_path[:-4] + ".png"
            mask_path = item.get('sphere_mask')
            depth_path = item.get('sphere_depth')
            gt_path = item.get('sphere_rgb')
            pose = item.get('camera_pose')
            fov = item.get('camera_angle_x')
            focal = fov2focal(fov, 1024)
            K = get_intrinsic_matrix(focal, 1024, 1024)
            image_name = item.get("room_id") + str(item.get("sphere_id")) + str(item.get("cam_id"))
            if self.use_wrapped_cond:
                sphere_path = item.get("wrapped_sphere_path")
        else:
            raise ValueError("Invalid val_set. Must be either '3dfront' or 'infinigen'.")

        example = {}


        image = Image.open(bg_path).convert("RGB")
        image = image.resize((1024, 1024))

        # Convert depth to a PIL image as well (if needed)

        example["image"] = image
        example["pose"] = pose
        example["K"] = K

        example["path"] = image_name
        example["pixel_values_path"] = gt_path
        example["masked_images_path"] = bg_path
        example["masks_path"] = mask_path
        example["depth_values_path"] = depth_path

        return example



class ALPLoader(Dataset):
    def __init__(self,
                 root=None,
                 num_samples=None,
                 res_threshold=((1024, 1024)),
                 apply_threshold=False,
                 **kwargs,
        ):
        super().__init__(**kwargs)
        self.root = root
        self.res_threshold = res_threshold
        self.apply_threshold = apply_threshold
        self.has_meta = False
        
        if self.root is not None:
            if not os.path.exists(self.root):
                raise Exception(f"Dataset {self.root} does not exist.") 
            
            dirs = natsorted(list(glob.glob(os.path.join(self.root, "*"))))
            self.scene_data = self._load_data_path(dirs)

    def _load_data_path(self, dirs):
        self.scene_names = [Path(dir).name for dir in dirs]

        scene_data = []
        for dir in dirs:
            pseudo_probe_dirs = natsorted(list(glob.glob(os.path.join(dir, "*"))))
            pseudo_probe_dirs = [dir for dir in pseudo_probe_dirs if "gt" not in dir]
            data = [os.path.join(dir, "images", "0.png") for dir in pseudo_probe_dirs]
            scene_data.append(data)

        scene_data = list(itertools.chain(*scene_data))
        return scene_data

class MultiIlluminationLoader(Dataset):
    def __init__(self,
                root, 
                mask_probe=True, 
                mask_boundingbox=False,
                **kwargs,
        ):
        """
        @params resolution (tuple): (width, height) - resolution of the image
        @params force_square: will add black border to make the image square while keeping the aspect ratio
        @params mask_probe: mask the probe with the mask in the dataset
        
        """
        super().__init__(**kwargs)
        self.root = root
        self.mask_probe = mask_probe
        self.mask_boundingbox = mask_boundingbox

        if self.root is not None:
            dirs = natsorted(list(glob.glob(os.path.join(self.root, "*"))))
            self.scene_data = self._load_data_path(dirs)

    def _load_data_path(self, dirs):
        self.scene_names = [Path(dir).name for dir in dirs]

        data = {}
        for dir in dirs:
            chrome_probes = natsorted(list(glob.glob(os.path.join(dir, "probes", "*chrome*.jpg"))))
            gray_probes = natsorted(list(glob.glob(os.path.join(dir, "probes", "*gray*.jpg"))))
            scenes = natsorted(list(glob.glob(os.path.join(dir, "dir_*.jpg"))))

            with open(os.path.join(dir, "meta.json")) as f:
                meta_data = json.load(f)
            
            bbox_chrome = meta_data["chrome"]["bounding_box"]
            bbox_gray = meta_data["gray"]["bounding_box"]

            mask_chrome = os.path.join(dir, "mask_chrome.png")
            mask_gray = os.path.join(dir, "mask_gray.png")

            scene_name = Path(dir).name
            data[scene_name] = {
                "scenes": scenes,
                "chrome_probes": chrome_probes,
                "gray_probes": gray_probes,
                "bbox_chrome": bbox_chrome,
                "bbox_gray": bbox_gray,
                "mask_chrome": mask_chrome,
                "mask_gray": mask_gray,
            }
        return data

    def _mask_probe(self, image, mask):
        """
        mask probe with a png file in dataset
        """
        image_anticheat = skimage.img_as_float(np.array(image))
        mask_np = skimage.img_as_float(np.array(mask))[..., None]
        image_anticheat = ((1.0 - mask_np) * image_anticheat) + (0.5 * mask_np)
        image_anticheat = Image.fromarray(skimage.img_as_ubyte(image_anticheat))
        return image_anticheat
    
    def _mask_boundingbox(self, image, bbox): 
        """
        mask image with the bounding box for anti-cheat
        """
        bbox = {k:int(np.round(v/4.0)) for k,v in bbox.items()}
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        image_anticheat = skimage.img_as_float(np.array(image))
        image_anticheat[y:y+h, x:x+w] = 0.5
        image_anticheat = Image.fromarray(skimage.img_as_ubyte(image_anticheat))
        return image_anticheat
        
    def __getitem__(self, scene_name):
        data = self.scene_data[scene_name]
        
        mask_chrome = Image.open(data["mask_chrome"])
        mask_gray = Image.open(data["mask_gray"])
        images = []
        for path in data["scenes"]:
            image = Image.open(path)
            if self.mask_probe:
                image = self._mask_probe(image, mask_chrome)
                image = self._mask_probe(image, mask_gray)
            if self.mask_boundingbox:
                image = self._mask_boundingbox(image, data["bbox_chrome"])
                image = self._mask_boundingbox(image, data["bbox_gray"])
                
            if self.force_square:
                image = pil_square_image(image, self.resolution)
            else:
                image = image.resize(self.resolution)
            images.append(image)

        chrome_probes = [Image.open(path) for path in data["chrome_probes"]]
        gray_probes = [Image.open(path) for path in data["gray_probes"]]
        bbox_chrome = data["bbox_chrome"]
        bbox_gray = data["bbox_gray"]
        
        return images, chrome_probes, gray_probes, bbox_chrome, bbox_gray

    
    def calculate_ball_info(self, scene_name):
        # TODO: remove hard-coded parameters
        ball_data = []
        for mtype in ['bbox_chrome', 'bbox_gray']:
            info = self.scene_data[scene_name][mtype]

            # x-y is top-left corner of the bounding box
            # meta file is for 4000x6000 image but dataset is 1000x1500
            x = info['x'] / 4
            y = info['y'] / 4
            w = info['w'] / 4
            h = info['h'] / 4

           
            # we scale data to 512x512 image 
            if self.force_square:
                h_ratio = (512.0 * 2.0 / 3.0) / 1000.0    #384 because we have black border on the top
                w_ratio = 512.0 / 1500.0
            else:
                h_ratio = self.resolution[0] / 1000.0
                w_ratio = self.resolution[1] / 1500.0
                
            x = x * w_ratio
            y = y * h_ratio
            w = w * w_ratio
            h = h * h_ratio

            if self.force_square:
                # y need to shift due to top black border
                top_border_height = 512.0 * (1/6)
                y = y + top_border_height


            # Sphere is not circle due to the camera perspective, Need future fix for this
            # For now, we use the minimum of width and height
            w = int(np.round(w))
            h = int(np.round(h))
            if w > h:
                r = h
                x = x + (w - h) / 2.0
            else:
                r = w 
                y = y + (h - w) / 2.0                
            
            x = int(np.round(x))
            y = int(np.round(y))
            
            ball_data.append((x, y, r))
        
        return ball_data
    
    def calculate_bbox_info(self, scene_name):
        # TODO: remove hard-coded parameters
        bbox_data = []
        for mtype in ['bbox_chrome', 'bbox_gray']:
            info = self.scene_data[scene_name][mtype]

            # x-y is top-left corner of the bounding box
            # meta file is for 4000x6000 image but dataset is 1000x1500
            x = info['x'] / 4
            y = info['y'] / 4
            w = info['w'] / 4
            h = info['h'] / 4

           
            # we scale data to 512x512 image 
            if self.force_square:
                h_ratio = (512.0 * 2.0 / 3.0) / 1000.0    #384 because we have black border on the top
                w_ratio = 512.0 / 1500.0
            else:
                h_ratio = self.resolution[0] / 1000.0
                w_ratio = self.resolution[1] / 1500.0
                
            x = x * w_ratio
            y = y * h_ratio
            w = w * w_ratio
            h = h * h_ratio

            if self.force_square:
                # y need to shift due to top black border
                top_border_height = 512.0 * (1/6)
                y = y + top_border_height


            w = int(np.round(w))
            h = int(np.round(h))             
            x = int(np.round(x))
            y = int(np.round(y))
            
            bbox_data.append((x, y, w, h))
        
        return bbox_data

    """
    DO NOT remove this!
    This is for evaluating results from Multi-Illumination generated from the old version
    """
    def calculate_ball_info_legacy(self, scene_name):
        # TODO: remove hard-coded parameters
        ball_data = []
        for mtype in ['bbox_chrome', 'bbox_gray']:
            info = self.scene_data[scene_name][mtype]

            # x-y is top-left corner of the bounding box
            # meta file is for 4000x6000 image but dataset is 1000x1500
            x = info['x'] / 4
            y = info['y'] / 4
            w = info['w'] / 4
            h = info['h'] / 4

            # we scale data to 512x512 image 
            h_ratio = 384.0 / 1000.0    #384 because we have black border on the top
            w_ratio = 512.0 / 1500.0
            x = x * w_ratio
            y = y * h_ratio
            w = w * w_ratio
            h = h * h_ratio

            # y need to shift due to top black border
            top_border_height = 512.0 * (1/8)

            y = y + top_border_height

            # Sphere is not circle due to the camera perspective, Need future fix for this
            # For now, we use the minimum of width and height
            r = np.max(np.array([w, h]))  

            x = int(np.round(x))
            y = int(np.round(y))
            r = int(np.round(r))

            ball_data.append((y, x, r))

        return ball_data