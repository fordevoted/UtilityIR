# Always Clear Days: Degradation Type and Severity Aware All-In-One Adverse Weather Removal
An official implement of the paper "Always Clear Days: Degradation Type and Severity Aware All-In-One Adverse Weather Removal"

[[Paper](https://arxiv.org/abs/2310.18293)]

[Yu-Wei, Chen](https://fordevoted.github.io), [Soo-Chang, Pei](https://scholar.google.com/citations?user=-JiGrnAAAAAJ&hl=zh-TW)


> **Abstract:**  All-in-one adverse weather removal is an emerging topic on image restoration, which aims to restore multiple weather degradation in an unified model, and the challenging are twofold. First, discovering and handling the property of multi-domain in target distribution formed by multiple weather conditions. Second, design efficient and effective operations for different degradation types. To address this problem, most prior works focus on the multi-domain caused by weather type. Inspired by inter\&intra-domain adaptation literature, we observed that not only weather type but also weather severity introduce multi-domain within each weather type domain, which is ignored by previous methods, and further limit their performance. To this end, we proposed a degradation type and severity aware model, called UtilityIR, for blind all-in-one bad weather image restoration. To extract weather information from single image, we proposed a novel Marginal Quality Ranking Loss (MQRL) and utilized Contrastive Loss (CL) to guide weather severity and type extraction, and leverage a bag of novel techniques such as Multi-Head Cross Attention (MHCA) and Local-Global Adaptive Instance Normalization (LG-AdaIN) to efficiently restore spatial varying weather degradation. The proposed method can significantly outperform the SOTA methods subjectively and objectively on different weather restoration tasks with a large margin, and enjoy less model parameters. Proposed method even can restore unseen domain combined multiple degradation images, and modulating restoration level.

## <a name="news"></a> 🆕 News
- **2023-10-27:** paper upload to arXiv.
- **2023-12-24:** Github repo setup. Implementation code and pretrained model will be released after paper acceptance.

## <a name="news"></a> 📖 Citation
If this work helps your research or work, please cite our paper
```
@article{chen2023always,
  title={Always Clear Days: Degradation Type and Severity Aware All-In-One Adverse Weather Removal},
  author={Chen, Yu-Wei and Pei, Soo-Chang},
  journal={arXiv preprint arXiv:2310.18293},
  year={2023}
}
```
## <a name="news"></a> 📜 License
This project is released under [Apache 2.0 license](LICENSE)


## <a name="news"></a> ✉️ Contact
Please free free let me know if you have any questions about this work by opening an issue or mail `210509fssh@gmail.com` `r09942066@ntu.edu.tw`
