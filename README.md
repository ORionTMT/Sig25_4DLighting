# Spatiotemporally Consistent Indoor Lighting Estimation with Diffusion Priors 
[[website](https://oriontmt.github.io/Sig25-4DLighting/)]

![teaser-figure](figures/teaser_more.png)
## TODO

- [x] Releasing the inference script
- [x] Releasing the chrome ball visualization script
- [ ] Uploading the complete training dataset and training script
- [ ] Uploading the benchmarking script

## Useage

### Setup enviroment


   
```bash
conda create -n 4DLighting python=3.11
conda activate 4DLighting

pip install -r requirements.txt
```

### Download the joint-predicting SD inpainting checkpoint from Google Drive
```bash
mkdir -p checkpoints
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1lPj_bhAFzKjdDTdM1eAL4Hx64EoHTtU_
mv Multi_Inpaint checkpoints/
```


### Run video demos

```bash
python run_video_demo.py \
        --ckpt_dir checkpoints/Multi_Inpaint \
        --video "Path to the video to be optimized" \
        --save_loc "Path to save the optimized results and video demo"
```



## Citations
```
@inproceedings{10.1145/3721238.3730749,
author = {Tong, Mutian and Wu, Rundi and Zheng, Changxi},
title = {Spatiotemporally Consistent Indoor Lighting Estimation with Diffusion Priors},
year = {2025},
isbn = {9798400715402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3721238.3730749},
doi = {10.1145/3721238.3730749},
booktitle = {Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers},
articleno = {107},
numpages = {11},
keywords = {Lighting estimation, diffusion models},
series = {SIGGRAPH Conference Papers '25}
}
```