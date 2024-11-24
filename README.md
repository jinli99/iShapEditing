# iShapEditing: Intelligent Shape Editing with Diffusion Models (PG 2024)

![Teaser image](images/pipeline.png)


Abstract: *Recent advancements in generative models have enabled image editing very effective with impressive results. By extending this progress to 3D geometry models, we introduce iShapEditing, a novel framework for 3D shape editing which is applicable to both generated and real shapes. Users manipulate shapes by dragging handle points to corresponding targets, offering an intuitive and intelligent editing interface. Leveraging the Triplane Diffusion model and robust intermediate feature correspondence, our framework utilizes classifier guidance to adjust noise representations during sampling process, ensuring alignment with user expectations while preserving plausibility. For real shapes, we employ shape predictions at each time step alongside a DDPM-based inversion algorithm to derive their latent codes, facilitating seamless editing. iShapEditing provides effective and intelligent control over shapes without the need for additional model training or fine-tuning. Experimental examples demonstrate the effectiveness and superiority of our method in terms of editing accuracy and plausibility.*

[Project Page](),   [Paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/cgf.15253)

## Install

Install PyTorch and other dependencies:

```
conda create --name iShapEditing python=3.10 -y
conda activate iShapEditing

conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install -r requirements.txt
```

Download the pre-trained Triplane Diffusion models from this [link](https://github.com/JRyanShue/NFD), and save them (chair, car, airplane) to `model` folder.

## GUI Usage
Simple demo
<p align="center">
<video src="images/gui.mp4" controls="controls" style="max-width: 100%; height: auto;">
    Your browser does not support the video tag.
</video>
</p>

Step 1: Run following instruction to setup the GUI interface:
```
python main.py
```
<p align="center">
    <img src="images/gui.png" alt="GUI Image" style="width:50%; height:auto;">
</p>

Step 2: Choose the model (chair/car/airplane).

Step 3: Get source shape:
- For the generated shape, click the `Create Mesh` button.
- For the real shape, click the `Load Mesh` button to load your shape. And then click the `Inversion` to do the triplane reconstruction and inversion. 

Step 4: Input handle points and target points. You have two ways to input the points:
- Push the `ctrl` keyboard and click left mouse to select point.
- Input the coordinate in the vector editor `XYZ` and then click the `Draw` button. Notice that when the last selection is the source points, then this XYZ value is the displacement of the next target point from last source point.

Step 5: Click the 'Start' button to start the dragging process.

## Citation
If you find our work useful in your research, please cite the following paper
```
@inproceedings{li2024ishapediting,
  title={iShapEditing: Intelligent Shape Editing with Diffusion Models},
  author={Li, Jing and Zhang, Juyong and Chen, Falai},
  booktitle={Computer Graphics Forum},
  pages={e15253},
  year={2024},
  organization={Wiley Online Library}
}
```