# T2S-Augmentation
本仓库开源了以下论文的代码：
- 标题：Target-to-Source Augmentation for Aspect Sentiment Triplet Extraction
- 作者：Yice Zhang, Yifan Yang, Meng Li, Bin Liang, Shiwei Chen, Ruifeng Xu
- 会议：EMNLP-2023 Main (Long)

# 工作简介

**背景**

方面情感三元组抽取（Aspect Sentiment Triplet Extraction, ASTE）是方面级情感分析（Aspect-Based Sentiment Analysis, ABSA）中的一个典型任务。该任务旨在从评论中抽取用户方面级别的情感和观点，以三元组的形式输出。其中，一个三元组由方面项（aspect term）、观点项（opinion term）、情感倾向（sentiment polarity）组成。作为一个细粒度情感分析任务，ASTE的数据标注代价较高；而标注数据的缺乏限制了现有方法的性能。

**传统的数据增强方法**

数据增强方法旨在根据现有的标注数据合成新的标注数据，是缓解数据稀缺问题的可行方法。传统的数据增强方法一般修改现有样例的输入文本，然后将修改后的文本和原有样例合并为新的样例。这种修改一般通过启发式的规则或者条件语言模型来实现。传统数据增强方法的一大问题是，难以在保证修改后句子和原标签一致的情况下，生成多样化的样本。

**Target-to-Source Augmentation**

本工作中，我们学习一个生成器直接根据标签和句法模板来生成新的句子。假设我们有一个足够强大的生成器，我们就可以混合来自不同样例的标签和句法模板，生成大量的多样化的样例。

正式地，我们的目标是学习一个生成器，其输入是一个标签和一个句法依存树，输出是一个句子。输出的句子应该满足三个要求：
- 流利性：应当是一个流利的句子，
- 一致性：应当包含给定的三元组而不引入新的三元组，
- 多样性：应当和现有的句子有显著的区别。

我们引入两个判别器来评估生成样本的流利性和一致性。

<div align="center"> <img src="https://github.com/HITSZ-HLT/T2S-Augmentation/assets/9134454/0b065e61-b622-4fa3-86a1-16323578f149" alt="target-to-source augmentation" width="45%" /></div>


**我们的方法**

- 模型架构：生成器是一个完整的Transformer结构（T5），流利度判别器和一致性判别器包含一个transformer编码器和一个二元分类器。
- 有监督学习：
  - 现有的标注数据集太小而无法训练可用的生成器和可靠的判别器。因此，我们基于现有的标注数据集构建了一个伪标注数据集。
  - 我们在伪标注数据集上优化生成器。
  - 对于流利度判别器，我们将该数据集上的句子作为流利的句子，然后将生成器生成的句子作为不流利的句子。
  - 对于一致性判别器，我们将该数据集上的样本作为一致的样本，然后使用beam search采样一些不一致的标签，从而合成不一致的样本。
- 强化学习：我们进一步使用一个强化学习的框架来根据判别器的反馈优化生成器。
  - 奖励计算：奖励包含三部分，流利度得分、一致性得分以及一个额外的长度惩罚。
  - 参数更新：我们是用PPO算法来进行生成器的参数更新。
- 样本合成及过滤
  - 我们从伪标签数据集中随机选择两个样本，然后将第一个样本的标签和第二个样本的句法依存树输入到生成器中生成句子。
    - 这里的两个样本需要有相同数目的三元组数目。
  - 我们使用判别器过滤合成样本中不流利和不一致的样本。
    
<div align="center"> <img src="https://github.com/HITSZ-HLT/T2S-Augmentation/assets/9134454/acc71dd1-6c48-477b-b218-80fd60cd9af1" alt="reinforcement learning framework" width="50%" /></div>

**实验结果**

我们在原标注数据集和增强后的数据集上运行了四个ASTE方法，结果如下。我们可以看到本文提出的增强方法在F1分数上取得了2.74%的平均提升。更多实验和分析请参见论文。

<div align="center"> <img src="https://github.com/HITSZ-HLT/T2S-Augmentation/assets/9134454/de73b43f-c18e-4254-bd08-711022bf89e0" alt="main result" width="70%" /></div>

# 运行代码

**环境配置**

- transformers==4.26.1
- torch==1.10.1
- pytorch-lightning==1.9.3
- rouge==1.0.1
- nltk==3.8.1
- sacrebleu==2.3.1
- spacy==3.5.0

**运行代码**

运行起来比较复杂，下面是14res上的一个参考。

```
chmod +x bash/*
bash/train_extractor.sh -c 0 -b extractor -d origin/14res
bash/do_extraction.sh -c 0 -m ./output/extraction/model/model/dataset=origin/14res,b=extractor,seed=42 -d data/unlabel/yelp2023/100k_1.json -o ./output/extraction/pseudo_labeled/yelp2023.json
python parsing.py --data_dir ./data/origin --dataset 14res --output_dir ./data/origin_syn
python parsing.py --data_dir ./output/extraction/pseudo_labeled --dataset yelp2023.json --output_dir ./output/extraction/pseudo_labeled_syn --main2
bash/train_generator.sh -c 0 -d ./data/origin_syn/14res -t ./output/extraction/pseudo_labeled/yelp2023.json -b generator
bash/build_fluency_dataset.sh -c 0 -d ./data/origin_syn/14res -t ./output/extraction/pseudo_labeled_syn/yelp2023.json -m ./output/generation/model/b=generator -o ./output/fluency_dataset_14res/
bash/build_alignment_dataset.sh -c 0 -d ./data/origin_syn/14res -t ./output/extraction/pseudo_labeled_syn/yelp2023.json -m ./output/extraction/model/model/dataset=origin/14res,b=extractor,seed=42 -o ./output/alignment_dataset_14res/
bash/train_fluency_discriminator.sh -c 0 -d ./output/fluency_dataset_14res/ -b fluency_model
bash/train_alignment_discriminator.sh -c 0 -d ./output/fluency_dataset_14res/ -b alignment_model
bash/ppo_tuning.sh -c 0 -b 14res -d data/origin_syn/14res -t ./output/extraction/pseudo_labeled_syn/yelp2023.json -g ./output/generation/model/b=generator -a ./output/alignment_model/model/b=alignment_model -f ./output/fluency_model/model/b=fluency_model -b ppo
bash/data_synthesis.sh -c 0 -b 14res_100k -g ./output/generation_ppo/model/b=ppo -e ./output/extraction/model/model/dataset=origin/14res,b=extractor,seed=42 -a ./output/alignment_model/model/b=alignment_model -f ./output/fluency_model/model/b=fluency_model -n 100000 -d data/origin_syn/14res -r output/extraction/pseudo_labeled/yelp2023.json
python data_filtering.py --origin_data_dir data/origin/14res --augmented_data_dir ./output/augmentation/14res_100k_42.json --output_dir ./output/augmentation_filtered/14res_5k_42 --k 5000
```
注：请事先解压`data/unlabeled`下的文件。

**生成的数据集**

我们将我们的方法在14res和14lap上应用，增强后的数据集可在`data/augmented`目录下获取。

  

