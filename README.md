<img width="737" alt="e02e6d28a5d659643e8aeb8d3075740" src="https://github.com/user-attachments/assets/8931d647-7837-4aeb-8c5b-fa077383a48c">

## 🌈
Training

for stage1 training (Spatiotemporal Projection Module)

```
sh shell/train/stage1_train.sh
```
for stage2 training (fine-tuning the vision encoder and LLM with LoRA)

```
sh shell/train/stage2_train.sh
```
## 📌 TODO
- ✅ Release the training code (stage1 and stage2)
- ✅ Release the evaluation code (stage1 and stage2)
- [ ] Release the inference code


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
