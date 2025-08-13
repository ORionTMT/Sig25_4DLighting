try:
    import cv2
except:
    pass
import numpy as np
from PIL import Image
from .ball_processor import get_ideal_normal_ball

def create_grid(image_size, n_ball, size):
    height, width = image_size
    nx, ny = n_ball
    if nx * ny == 1:
        grid = np.array([[(height-size)//2, (width-size)//2]])
    else:
        height_ = np.linspace(0, height-size, nx).astype(int)
        weight_ = np.linspace(0, width-size, ny).astype(int)
        hh, ww = np.meshgrid(height_, weight_)
        grid = np.stack([hh,ww], axis = -1).reshape(-1,2)

    return grid

class MaskGenerator():
    def __init__(self, cache_mask=True):
        self.cache_mask = cache_mask
        self.all_masks = []

    def clear_cache(self):
        self.all_masks = []

    def retrieve_masks(self):
        return self.all_masks

    def generate_grid(self, image, mask_ball, n_ball=16, size=128):
        ball_positions = create_grid(image.size, n_ball, size)
        # _, mask_ball = get_normal_ball(size)
        
        masks = []
        mask_template = np.zeros(image.size)
        for x, y in ball_positions:
            mask = mask_template.copy()
            mask[y:y+size, x:x+size] = 255 * mask_ball
            mask = Image.fromarray(mask.astype(np.uint8), "L")
            masks.append(mask)

            # if self.cache_mask:
            #     self.all_masks.append((x, y, size))
        
        return masks, ball_positions

    def generate_single(self, image, mask_ball, x, y, size):
        w,h = image.size # numpy as (h,w) but PIL is (w,h)
        mask = np.zeros((h,w))
        mask[y:y+size, x:x+size] = 255 * mask_ball
        mask = Image.fromarray(mask.astype(np.uint8), "L")

        return mask
    def generate_grid_masks(self, image, mask_ball, size):
        """
        在图像上生成九个均匀分布的 mask。
        
        @params:
            - image: 输入图像（PIL.Image）。
            - mask_ball: 单个法线球 mask。
            - size: mask 的大小（正方形区域的边长）。
        @return:
            - grid_mask: 包含所有 mask 的图像。
        """
        w, h = image.size  # 图像宽高 (w, h)，PIL 是 (w, h)
        grid_mask = np.zeros((h, w))  # 初始化整个图像的 mask

        # 计算 1/4, 2/4, 3/4 的位置
        #x_positions = [w // 3-20, w // 2, 2 * w // 3+20]
        #y_positions = [h // 3-20, h // 2, 2 * h // 3+20]
        #x_positions = [w // 8, 3* w // 8, 5* w // 8, 7* w // 8]
        #y_positions = [h // 8, 3* h // 8, 5* h // 8, 7* h // 8]
        #x_positions = [72, 195, 318, 440]
        #y_positions = [72, 195, 318, 440]
        x_positions = [160, 350]
        y_positions = [160, 350]

        # 遍历格点
        for cx in x_positions:
            for cy in y_positions:
                # 计算当前 mask 的左上角 (x, y)
                x = int(cx - size // 2)
                y = int(cy - size // 2)

                # 检查边界，确保 mask 不越界
                x = max(0, min(w - size, x))
                y = max(0, min(h - size, y))

                # 将 mask_ball 填入对应位置
                grid_mask[y:y + size, x:x + size] += 255 * mask_ball  # 叠加

        # 将 numpy 数组转换为 PIL 图像
        grid_mask = Image.fromarray(grid_mask.astype(np.uint8), "L")
        return grid_mask


    def generate_best(self, image, mask_ball, size):
        w,h = image.size # numpy as (h,w) but PIL is (w,h)
        mask = np.zeros((h,w))

        (y, x), _ = find_best_location(np.array(image), ball_size=size)
        mask[y:y+size, x:x+size] = 255 * mask_ball
        mask = Image.fromarray(mask.astype(np.uint8), "L")

        return mask, (x, y)
        

def get_only_high_freqency(image: np.array):
    """
    Get only height freqency image by subtract low freqency (using gaussian blur)
    @params image: np.array - image in RGB format [h,w,3]
    @return high_frequency: np.array - high freqnecy image in grayscale format [h,w] 
    """
 
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  
    # Subtract low freqency from high freqency
    kernel_size = 11  # Adjust this according to your image size
    high_frequency = gray - cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
 
    return high_frequency

def find_best_location(image, ball_size=128):
    """
    Find the best location to place the ball (Eg. empty location)
    @params image: np.array - image in RGB format [h,w,3]
    @return min_pos: tuple - top left position of the best location (the location is in "Y,X" format)
    @return min_val: float - the sum value contain in the window
    """
    local_variance = get_only_high_freqency(image)
    qsum = quicksum2d(local_variance)
 
    min_val = None
    min_pos = None
    k = ball_size
    for i in range(k-1, qsum.shape[0]):
        for j in range(k-1, qsum.shape[1]):
            A = 0 if i-k < 0 else qsum[i-k, j]
            B = 0 if j-k < 0 else qsum[i, j-k] 
            C = 0 if (i-k < 0) or (j-k < 0) else qsum[i-k, j-k]
            sum = qsum[i, j] - A - B + C
            if (min_val is None) or (sum < min_val):
                min_val = sum
                min_pos = (i-k+1, j-k+1) # get top left position
 
    return min_pos, min_val

def quicksum2d(x: np.array):
    """
    Quick sum algorithm to find the window that have smallest sum with O(n^2) complexity
    @params x: np.array - image in grayscale [h,w]
    @return q: np.array - quick sum of the image for future seach in find_best_location [h,w]
    """
    qsum = np.zeros(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            A = 0 if i-1 < 0 else qsum[i-1, j]
            B = 0 if j-1 < 0 else qsum[i, j-1] 
            C = 0 if (i-1 < 0) or (j-1 < 0) else qsum[i-1, j-1]
            qsum[i, j] = A + B - C + x[i, j]

    return qsum