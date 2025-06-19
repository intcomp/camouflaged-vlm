# Open-Vocabulary Camouflaged Object Segmentation with Cascaded Vision Language Models



![abstract_img](.\abstract_img\overview.jpg)

## Prepare work

1. sam-vit-h-pth: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

2. alpha-clip-best-model.pth and best-model.pth: 
   link: https://pan.baidu.com/s/1q2XueAeRyFBend4ilGIQ2w?pwd=siy7  password: siy7 



Put the 'sam-vit-h-pth'  and 'alpha-clip-best-model.pth' in the folder *pretrained*.

Put the 'best-model.pth' in the folder *best_model_pth*.



## Test

```bash
python test_ovsam_maskdecoder1_edge.py
```



