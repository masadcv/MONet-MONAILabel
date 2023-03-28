# Adaptive Multi-scale Online Likelihood Network for AI-assisted Interactive Segmentation (MONet)
This repository provides source code for MONet, a multi-scale online likelihood method for scribble-based interactive segmentation. If you use this code, please cite the following paper:

> Asad, Muhammad, Helena Williams, Indrajeet Mandal, Sarim Ather, Jan Deprest, Jan D'hooge, and Tom Vercauteren 
>"[Adaptive Multi-scale Online Likelihood Network for AI-assisted Interactive Segmentation](https://arxiv.org/abs/2303.13696)" 
>arXiv preprint arXiv:2303.13696 (2023).
#  Introduction
Existing interactive segmentation methods leverage automatic segmentation and user interactions for label refinement, significantly reducing the annotation workload compared to manual annotation. However, these methods lack quick adaptability to ambiguous and noisy data, which is a challenge in CT volumes containing lung lesions from COVID-19 patients. In this work, we propose an adaptive multi-scale online likelihood network (MONet) that adaptively learns in a data-efficient online setting from both an initial automatic segmentation and user interactions providing corrections. We achieve adaptive learning by proposing an adaptive loss that extends the influence of user-provided interaction to neighboring regions with similar features. In addition, we propose a data-efficient probability-guided pruning method that discards uncertain and redundant labels in the initial segmentation to enable efficient online training and inference. Our proposed method was evaluated by an expert in a blinded comparative study on COVID-19 lung lesion annotation task in CT. Our approach achieved 5.86% higher Dice score with 24.67% less perceived NASA-TLX workload score than the state-of-the-art.

![monet-flowchart](https://raw.githubusercontent.com/masadcv/MONet-MONAILabel/main/data/model-MONet.png)

The flowchart above shows (a) training and inference of MONet using adaptive loss and probability-guided pruning; (b) architecture of our multi-scale online likelihood network (MONet).

Further details about MONet can be found in our paper linked above.

# Methods Included
In addition to MONet, we include comparison methods used in our paper. These are summarised in table below:
| Method Name                | Description                      |
|----------------------------|----------------------------------|
| MONet + GraphCut          | MONet (proposed) based likelihood inference               |
| MONet-NoMS + GraphCut      | MONet-NoMS without multi-scale features for likelihood inference               |
| ECONet + GraphCut          | ECONet [1] based likelihood inference               |
| ECONet-Haar + GraphCut     | ECONet with Haar-Like [1] features for likelihood inference             |
| Interactive + GraphCut       | Interactive Graphcut based on [2] |

[1] Asad, Muhammad, Lucas Fidon, and Tom Vercauteren. "ECONet: Efficient convolutional online likelihood network for scribble-based interactive segmentation." International Conference on Medical Imaging with Deep Learning. PMLR, 2022.

[2] Boykov, Yuri Y., and M-P. Jolly. "Interactive graph cuts for optimal boundary & region segmentation of objects in ND images." Proceedings eighth IEEE international conference on computer vision. ICCV 2001. Vol. 1. IEEE, 2001.

# Installation Instructions
MONet is implemented using [MONAI Label](https://github.com/Project-MONAI/MONAILabel), which is an AI-Assisted tool for developing interactive segmentation methods. We provide the MONet MONAI Label app that can be run with following steps:

- Clone MONet repo: `git clone https://github.com/masadcv/MONet-MONAILabel`
- Install requirements (recommended in new virtualenv): `pip install -r requirements.txt`
- Download and install **3D Slicer** from: [https://download.slicer.org/](https://download.slicer.org/)
- Install **MONAILabel extension** from `plugins` folder as this is using an older version of MONAILabel. For further help follow steps from: TODO Add tutorial to setup plugin

More detailed documentation on setting up MONAI Label can be found at: [https://docs.monai.io/projects/label/en/latest/installation.html](https://docs.monai.io/projects/label/en/latest/installation.html)

# Pretrained Model Download
TODO: Add links to pretrained models

# Running the MONet App
The MONet MONAI Label App runs as MONAI Label server and connects to a MONAI Label client plugin (3D Slicer/OHIF)

## Server: Running MONet Server App
MONet MONAI Label server can be started using MONAI Label CLI as:
```
monailabel start_server --app /path/to/this/github/clone --studies /path/to/dataset/images
```

e.g. command to run with sample data from root of this directory
```
monailabel start_server --app . --studies ./data/
```

> By default, MONAI Label server for MONet will be up and serving at: https://127.0.0.1:8000

## Client: Annotating CT Volumes using MONet on Client Plugin
On the client side, run slicer and load MONAILabel extension:
- Load MONet server at: https://127.0.0.1:8000
- Click Next Sample to load an input CT volume
- (optional) Click run under Auto Segmentation to get initial segmentation from UNet
- Scribbles-based interactive segmentation functionality is inside Scribbles section
- To add scribbles select scribbles-based interactive segmentation method then Painter/Eraser Tool and appropriate label Foreground/Background
- Painting/Erasing tool will be activated, add scribbles to each slice/view
- Once done, click Update to send scribbles to server for applying the selected method
- Iterate as many times as needed, then click submit to save final segmentation

<!-- A demo video showing this usage can be found here: [https://www.youtube.com/watch?v=kVGf5QQxSfc](https://www.youtube.com/watch?v=kVGf5QQxSfc) -->

<!-- ![econet-preview](./data/econet_preview.png) -->

# Citing MONet
Pre-print of MONet can be found at: [Adaptive Multi-scale Online Likelihood Network for AI-assisted Interactive Segmentation](https://arxiv.org/abs/2303.13696)

If you use MONet in your research, then please cite:

> Asad, Muhammad, Helena Williams, Indrajeet Mandal, Sarim Ather, Jan Deprest, Jan D'hooge, and Tom Vercauteren 
>"[Adaptive Multi-scale Online Likelihood Network for AI-assisted Interactive Segmentation](https://arxiv.org/abs/2303.13696)" 
>arXiv preprint arXiv:2303.13696 (2023).

BibTeX:
```
@article{asad2023monet,
  title={Adaptive Multi-scale Online Likelihood Network for AI-assisted Interactive Segmentation},
  author={Asad, Muhammad and Williams, Helena and  Mandal, Indrajeet and Ather, Sarim and Deprest, Jan and D'hooge, Jan and Vercauteren, Tom},
  journal={arXiv preprint arXiv:2303.13696},
  year={2023}
}
```
