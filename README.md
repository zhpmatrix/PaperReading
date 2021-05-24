277.《Lifelong Learning based Disease Diagnosis on Clinical Notes》，腾讯天衍实验室的工作，TODO

276.《PRGC: Potential Relation and Global Correspondence Based Joint Relational Triple Extraction》，腾讯天衍实验室的工作，TODO

275.《MIE: A Medical Information Extractor towards Medical Dialogues》

annotate online medical consultation dialogues in a window-sliding style.

274.《A Survey of Data Augmentation Approaches for NLP》

比较新的NLP数据增强文章，按照任务类型划分增强的方式。

273.《HyperCore: Hyperbolic and Co-graph Representation for Automatic ICD Coding》， ACL2021

ICD编码映射，医疗NLP的特色任务。

ICD的特点：

（1）层次

（2）共现(因果)

主要方法：GCN的应用


272.《Few-Shot Named Entity Recognition: A Comprehensive Study》，Jiawei Han组的工作

NER中的少样本问题，三种解决方案：

（1）基于proto的few-shot learning方法（此前在研究文本分类的few-shot问题时，该方向上的工作也一直比较受欢迎）

（2）带噪音，有监督预训练

（3）伪标签，自训练


271.《Summarizing Medical Conversations via Identifying Important Utterances》，COLING2020

主要内容：从医疗问答中抽取摘要。

数据：从春雨医生爬取

方法：抽取式摘要

270.《Enquire One’s Parent and Child Before Decision_ Fully Exploit Hierarchical Structure for Self-Supervised Taxonomy Expansion》，腾讯，刘邦

**分类树扩展**的工作，用于腾讯的疫情问答场景。刘邦的博士论文可以一读。

269.Federated Learning

《Privacy-Preserving Technology to Help Millioins of People_Federated Prediction Model for Stroke Prevention》

FL使用传统模型，也是目前主要做的工作

《Empirical Studies of Institutional Federated Learning For Natural Language Processing》

FL使用TextCNN的经验性工作

《FedED: Federated Learning via Ensemble Distillation for Medical Relation Extraction》

内容：FL应用于医疗关系抽取

结果：实现了隐私保护，但是指标下降

核心：在通信，不在计算

科普：[《Introduction to FL》](https://www.youtube.com/watch?v=STxtRucv_zo)，本质上还是分布式学习的一种。

结论：**除非必要，否则目前在工业界推进的ROI应该不算高。不单纯是一个算法问题，还是一个架构问题。但是在医疗行业目前现状下（数据孤岛现象），仍有必要关注**

268.《MedDG: A Large-scale Medical Consultation Dataset for Building Medical Dialogue System》，**Xiaodan Liang**等

构建了一个中文医学对话数据集，特点是：标注了每个对话可能涉及的实体。

基于该数据集，定义了两个任务：

（1）next entity prediction。文章中用multi-label classification的方式实现

（2）doctor response generation。标准的文本生成类任务+融合任务（1）中的实体信息（最简单的方式：直接concat实体）

其他：ICLR2021要基于该数据集举办一个比赛，可以关注。

想法：其实是对生成领域强化对实体信息的利用。传统做生成的同学有一些对应的方式强化对实体信息的利用。不过，文章中的建模方式更偏intent识别。

《MedDialog: Large-scale Medical Dialogue Datasets》，EMNLP2020，这篇工作也是构建了一个中文医疗对话数据集，不过没有实体信息。

267.《BioBERT：a pre-trained biomedical language representation model for biomedical text mining》

预训练任务没有做任何改进，但是在下游的三个理解任务上均取得了提升，比较适合工业界操作的工作。

![img267](https://ftp.bmp.ovh/imgs/2021/03/20a8a5f451f89ef3.png)

补充：《Conceptualized Representation Learning for Chinese Biomedical Text Mining》，阿里巴巴，张宁豫

![img267+](https://ftp.bmp.ovh/imgs/2021/03/5d1227f4839fe5de.png)



266.《Building Watson：An Overview of the DeepQA Project》，2011年，IBM Watson的DeepQA项目，具体实现细节

讨论了架构和工程实现的问题，其中的特色在于对**证据**的重视。

265.《Strategies For Pre-training Graph Neural Networks》，ICLR2020

主要内容：预训练图的工作

motivation：node的pretrain和graph的pretrain都要；之前的一些工作只考虑node或者graph的单个类型的pretrain

训练任务：

（1）node：context graph的定义，学习context；attribute mask任务
（2）graph：supervised graph-level properties prediction + structural graph similarity prediction

基础模型： GIN

直观感受：中规中矩；目前还没看到预训练图的工作应用于电商领域等

264.《A User-Centered Concept Mining System for Query and Document Understanding at Tencent》，KDD2019

腾讯刘邦的工作，刘邦的博士论文也有share，主要做概念挖掘，偏向于工程系统的工作。[相关文章](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650419989&idx=2&sn=ffd0e6f04c81465ca263666a97736418&chksm=becdb54f89ba3c590023cd593456f2252ffa4d3b0eba5923449c251fb874e315ab25f5e24bd2&scene=0&xtrack=1&key=f1baf9836e24d95b25894f5ed75172f24088bf020a51854847f3b766f363dcada55a2bb32d0b7035a4e950a75d243fa430b8ad5afc8fa5980995ce0205499d08f3575c568de5fd4256087f072173f32af6ccdad480c4a5c1f32f66cdb7ee4b734c6cb8adb7444d941bc69bd6d3640ecb06673e5d2678be04dd7a85447a829c28&ascene=0&uin=MTg2NTIxNzUxNw%3D%3D&devicetype=iMac+MacBookAir7%2C2+OSX+OSX+10.15.2+build(19C57)&version=11020201&lang=zh_CN&exportkey=Ax5WXz28zKfYGuYgxDzoJso%3D&pass_ticket=RSqRY9vnIzdQMSWnSw37xTPvHc6zF8ityHYJXITPs%2FtXSrUn%2FMohvluSBUnn1Zxt&wx_header=0)

263.《Read, Retrospect, Select: An MRC Framework to Short Text Entity Linking》

和《AutoRegressive Entity Retrieval》，ICLR2021一块儿读。

262.《CRSLab：An Open-Source Toolkit for Building Conversational Recommender System》

CRS系统的设定：

![img_262](https://ftp.bmp.ovh/imgs/2021/01/d3be7c25a4f1419c.png)


261.《Open Domain Event Extraction Using Neural Latent Variable Models》

开放域的事件抽取。（个人对隐变量模型不是很了解）

260.《Billion-scale Commodity Embedding for E-commerce Recommendation in Alibaba》，KDD2018

![260_1_img](https://ftp.bmp.ovh/imgs/2021/01/db47ef98a5f065f4.png)

![260_2_img](https://ftp.bmp.ovh/imgs/2021/01/fbc61749b4d401cf.png)

259.《AutoRegressive Entity Retrieval》，ICLR2021

用生成的方式（seq2seq）做el，entity disambiguation， page retrieval任务。将传统分类任务转化为一个生成任务是问题解决范式的转变，很有意思的工作。在自己的博客，[MRC is all you need?](https://zhpmatrix.github.io/2020/05/07/mrc-is-all-you-need/)中讨论了将很多经典NLP任务用MRC的方式来做。

258.《Metapath-guided Heterogeneous Graph Neural Network for Intent Recommendation》，KDD2019

基于user-item-query构建的图，三种类型的边：search,click, guide，用于淘宝意图检测。

257.《Heterogeneous Graph Attention Network》，WWW2019

异构图+gat，文章写作思路很赞。目前，我们正尝试将该工作用于销量预测与归因分析。

256.《POG：Personalized Outfit Generation for Fashion Recommendation ai Alibaba iFashion》，KDD2019

阿里dida平台建设，compatibility+personality都要考察。

相关PR稿(dida也是从luban演化而来)：https://hackernoon.com/finding-the-perfect-outfit-with-alibabas-dida-ai-assistant-71ba7c9e8cfa

255.《Spam Review Detection with Graph Convolutional Networks》,CIKM2018 Best Paper

重新翻开这篇文章，方法如下：

![img_255](https://ftp.bmp.ovh/imgs/2020/12/42595b8c0cace4ad.png)

文章要解决的问题是垃圾评论检测，构建了两个图。第一个图：用户-评论-商品图，是异构图；第二个图：评论-评论图，是同构图。分别用异构GCN和GCN学到各自的表征，做节点分类工作。

整体上，文章的思路和这篇《Abusive Language Detection with Graph Convolutional Networks》非常相似，但是二者都没有互相引用。这篇文章做的是Tweet分类（三分类），分别构建两个图。第一个图：用户-用户的同构图；第二个图：用户-Tweet的异构图。针对同构图，用node2vec去学到表征（node2vec不是仅仅适用于同构图，不过效果需要考察）；针对异构图，用gcn去学到表征。表征组合（embedding+n-gram）+分类器做节点分类。

对比二者，整体上的技术思路相似，不过显然后者在图构建上更加的自然。

254.《Graph Neural Networks：Taxonomy, Advances and Trends》，最新的GNN相关的综述文章

253.《Understanding Image Retrieval Re-Ranking：A Graph Neural Network Persperctive》

有意思的工作，作者提到：

  （1）Re-ranking can be reformulated as a high-parallelism Graph Neural Network (GNN) function.

  （2）On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms with one K40m GPU.

252.《Why Are Deep Learning Models Not Consistently Winning Recommender Systems Competitions Yet?》，RecSys2020

非常棒的文章，多年以前自己就很好奇了。

251.《Enriching Pre-trained Language Model with Entity Information for Relation Classiﬁcation》

![img_251](https://ftp.bmp.ovh/imgs/2020/11/fd20de4ee9f75cfc.png)

250.《Diverse, Controllable, and Keyphrase-Aware:
A Corpus and Method for News Multi-Headline Generation》

新闻标题生成

249.《CharBERT: Character-aware Pre-trained Language Model》

248.《
