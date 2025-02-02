<div align="center">
  
<a href="https://arxiv.org/abs/2411.17221"><img src="https://img.shields.io/badge/Arxiv-2411:03795-red"></a>
<a href="https://pan.baidu.com/s/1A521vyf3NuJ6Ptl0KzM0aA?pwd=e2nk"><img src="https://img.shields.io/badge/Dataset-Download-red?logo=googlechrome&logoColor=red"></a>
</div>
<div>

  <h1>AIGV-Assessor: Benchmarking and Evaluating the Perceptual Quality of Text-to-Video Generation with LMM</h1> 
</div>

<img width="width: 80%" alt="e02e6d28a5d659643e8aeb8d3075740" src="https://github.com/user-attachments/assets/8931d647-7837-4aeb-8c5b-fa077383a48c">

## :fire: AIGVQA-DB
The AIGVQA-DB is available at 百度网盘 [https://pan.baidu.com/s/1A521vyf3NuJ6Ptl0KzM0aA?pwd=e2nk] 提取码: e2nk 

## 🛠️ Installation

Clone this repository:
```
git clone https://github.com/wangjiarui153/AIGV-Assessor.git
```
Create a conda virtual environment and activate it:
```
conda create -n aigvassessor python=3.9 -y
conda activate aigvassessor
```
Install dependencies using requirements.txt:
```
pip install -r requirements.txt
```

## 🌈 Training

for stage1 training (Spatiotemporal Projection Module)

```
sh shell/train/stage1_train.sh
```
for stage2 training (Fine-tuning the vision encoder and LLM with LoRA)

```
sh shell/train/stage2_train.sh
```
## 🌈 Evaluation

for stage1 evaluation (Text-based quality levels)

```
sh shell/eval/stage1_eval.sh
```
for stage2 evaluation (Scores from 4 perspectives)

```
sh shell/eval/stage2_eval.sh
```

## 📌 TODO
- ✅ Release the training code (stage1 and stage2)
- ✅ Release the evaluation code (stage1 and stage2)
- ✅ Release the AIGVQA-DB


## 📧 Contact
If you have any inquiries, please don't hesitate to reach out via email at `wangjiarui@sjtu.edu.cn`

## 🎓Citations

If you find AIGV-Assessor is helpful, please cite:

```bibtex
@misc{wang2024aigvassessorbenchmarkingevaluatingperceptual,
      title={AIGV-Assessor: Benchmarking and Evaluating the Perceptual Quality of Text-to-Video Generation with LMM}, 
      author={Jiarui Wang and Huiyu Duan and Guangtao Zhai and Juntong Wang and Xiongkuo Min},
      year={2024},
      eprint={2411.17221},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17221}, 
}
```
