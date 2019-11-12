# ok-emotion-analyzer
![](https://img.shields.io/badge/python-3.x-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.12-brightgreen.svg) ![](https://img.shields.io/badge/version-1.0.0-brightgreen.svg) ![](https://img.shields.io/badge/license-MIT-000000.svg)

一个基于cnn+adjacent feature结构、使用tensorflow实现的情感分析模型。基于卷积神经网络（CNN）的方法在情感分类任务中已经取得了不错的效果，此类模型使用词向量作为网络的输入，但是在卷积过程中每个词向量只能表征单个单词，并不蕴含上下文信息，这不利于信息传递的连续性，并且卷积操作在局部范围内可能会打乱词向量的序列性。本项目中提出的临近特征（Adjacent Feature）很好的解决了这个问题，该机制可同时考虑到前后词语的语义，并一同代入下一轮卷积计算中。



## 1、模型介绍

在卷积过程中，对于某个词语![img](https://ae01.alicdn.com/kf/Haf41e1a127584432960e18947ae592ac0.png)采取3种不同的邻近特征附加策略：左邻近特征（Left Adjacent Feature, LAF）、右邻近特征（Right Adjacent Feature, RAF）、左右邻近特征（Left Right Adjacent Feature, LRAF），它们的作用都是为了增强词语在特定上下文中所表达的语义。如果使用LAF，则卷积过程中使用词向量![img](https://ae01.alicdn.com/kf/H4d8fa23f241d4970a2e1695a9f9eee83v.png)，即从第一个词向量开始，每一个词向量都会与其左侧的词向量特征进行合并操作，这样做的好处是前一个词向量的信息可以传递给下一个词向量，因此下一个词向量在参加卷积操作时可以提供相对充足的情感信息供卷积层提取到有效的特征。![img](https://ae01.alicdn.com/kf/H7a8926a583da409ea78377ebff499defy.png)中每一个词向量与其右侧的词向量特征合并，![img](https://ae01.alicdn.com/kf/H96e11e0a9d8048dab0e119637689c027Q.png)中每一个词向量与其左、右侧的词向量特征合并。这三种策略的直观描述如下图所示，其中![img](https://ae01.alicdn.com/kf/H5f9a120c76864a70b70ba71a578a821fn.png)表示零向量。



![](https://ae01.alicdn.com/kf/H9beb2ef26ef14a0c94eeb9087c0b0a42l.png)



假设输入层句子![img](https://ae01.alicdn.com/kf/Hb480eefc9dda4359bd715fa16ba8e741n.png)的长度为![img](https://ae01.alicdn.com/kf/H73ea3d1db3a943529ccab7d412b097c1t.png)，![img](https://ae01.alicdn.com/kf/Hfcf898778114473595b7d774f4290663f.png)表示![img](https://ae01.alicdn.com/kf/Hd52894d66ad14bdfa5a0b5e58007ac14M.png)维的词向量，对应句子![img](https://ae01.alicdn.com/kf/Hb480eefc9dda4359bd715fa16ba8e741n.png)中的第![img](https://ae01.alicdn.com/kf/H88efc9fd0a6f4d3abb7527b4ebd0d548a.png)个词语，那么句子![img](https://ae01.alicdn.com/kf/Hb480eefc9dda4359bd715fa16ba8e741n.png)就可以表示成：

![img](https://ae01.alicdn.com/kf/He800529dafe44cb2aeccf0e274e554a58.png)

其中![img](https://ae01.alicdn.com/kf/Hc8f2cfe705914414a85a48e0be0b684bg.png)表示串联操作，换句话说，![img](https://ae01.alicdn.com/kf/H10222c34e9684e2dab9e69947c3fda0ah.png)表示词向量![img](https://ae01.alicdn.com/kf/H267a3172267743c2b4330c3daf2f77afz.png)的串联。如果使用邻近特征附加策略，则句子![img](https://ae01.alicdn.com/kf/Hb480eefc9dda4359bd715fa16ba8e741n.png)会有另外3种表示形式（图1中使用的是![img](https://ae01.alicdn.com/kf/Hbae37384c7124482967c3d6435c66554X.png)）：

![](https://ae01.alicdn.com/kf/Hb3792a4ba372453eb50f4dd8e0755912p.png)

​							![](https://ae01.alicdn.com/kf/H2b61a7745d0c4af286df144fa79befa7M.png)

![](https://ae01.alicdn.com/kf/H8ea7a30db9af4d6f9f33b46b6b1954bco.png)

其中，运算符![img](https://ae01.alicdn.com/kf/Hd62728527d854926b255a6d772744e2bo.png)为两个向量的合并符号，例如向量![img](https://puui.qpic.cn/fans_admin/0/3_1574395846_1571175447782/0)，向量![img](https://ae01.alicdn.com/kf/Ha9eaa17f3d9848e3a40ec4788e6643c3v.png)，那么有![img](https://ae01.alicdn.com/kf/H4dde4e4636274f2bb8668080bc15c028s.png)。整个模型如下图所示（这里假设是2分类）。



![](https://ae01.alicdn.com/kf/Hef10d96eff294000b78be3b4edc0349et.png)				



## 2、项目说明

目录TF_WORD2VEC_CNN_C2是2分类模型，TF_WORD2VEC_CNN_C3是3分类模型，其中均包含get_corpus_info.py(获取预料基本信息)、data_helper.py(数据预处理)、cnn_model.py(模型结构)、train_dev_test.py(训练、验证、测试)文件，以TF_WORD2VEC_CNN_C3中该文件为例，可配置参数在train_dev_test.py文件顶部：

```
## 配置说明
# Data path and parameter
# 负面语料文件路径
tf.flags.DEFINE_string("positive_data_file", "./data/1.txt", "Data source for the positive data.")
# 中性语料文件路径
tf.flags.DEFINE_string("neutral_data_file", "./data/0.txt", "Data source for the neutral data.")
# 正面语料文件路径
tf.flags.DEFINE_string("negtive_data_file", "./data/-1.txt", "Data source for the negative data.")
# 使用word2vec训练的词向量路径
tf.flags.DEFINE_string("w2vModelPath", "/home/lvchao/vectors/vectors_300.bin", "the model of word2vec")
# 语料中最大的句子长度
tf.flags.DEFINE_integer("sequence_length", 102, "the max of words in a sentence")

# Model Hyperparameters
# 临近特征附加策略，可选NC、LC、RC、LRC，默认NC
tf.flags.DEFINE_string("cxt_type", "NC", "Adjacent feature strategy (default: NC)")
# 词向量长度，默认300
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
# 使用3种过滤器以及尺寸
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
# 过滤器数量
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
# dropout比例
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# L2正则参数
tf.flags.DEFINE_float("l2_reg_lambda", 1.0, "L2 regularizaion lambda (default: 1.0)")

# Training parameters
# 批处理大小
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
# 训练多少轮epoch
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

# Misc Parameters
# 如果指定的GPU设备不存在，允许TF自动分配设备
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# 是否打印设备分配日志
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


## 运行程序
# 配置好以上默认参数后，即可开始训练、测试，运行train_dev_test.py文件时亦可从控制台重新指定各个参数，比如指定临近特征附加策略为'不附加'，此时模型结构实际是基本CNN模型
python train_dev_test.py --cxt_type="NC"
```



## 3、测试结果

模型对比实验在COAE2014上的查准率、召回率、F1值、准确率：

<table style="width:100%;"><tr ><td  valign=center  nowrap  rowspan=2   >模型</td><td   valign=center  nowrap  colspan=3   >负面</td><td   valign=center  nowrap  colspan=3   >正面</td><td   valign=center  nowrap  rowspan=2   >准确率</td></tr><tr  ><td   valign=center  nowrap   >查准率</td><td   valign=center  nowrap   >召回率</td><td   valign=center  nowrap   >F1<font face="华文中宋" >值</font></td><td   valign=center  nowrap   >查准率</td><td   valign=center  nowrap   >召回率</td><td   valign=center  nowrap   >F1<font face="华文中宋" >值</font></td></tr><tr  ><td  valign=center  nowrap   >CNN</td><td   valign=center  nowrap   >87.30%</td><td   valign=center  nowrap   >82.72%</td><td   valign=center  nowrap   >84.94%</td><td   valign=center  nowrap   >85.75%</td><td   valign=center  nowrap   >89.63%</td><td   valign=center  nowrap   >87.65%</td><td   valign=center  nowrap   >86.43%</td></tr><tr  ><td  valign=center  nowrap   >LAF-CNN</td><td   valign=center  nowrap   >87.38%</td><td   valign=center  nowrap   >85.49%</td><td   valign=center  nowrap   >86.43%</td><td   valign=center  nowrap   >87.73%</td><td   valign=center  nowrap   >89.36%</td><td   valign=center  nowrap   >88.54%</td><td   valign=center  nowrap   >87.57%</td></tr><tr  ><td  valign=center  nowrap   >RAF-CNN</td><td   valign=center  nowrap   >88.20%</td><td   valign=center  nowrap   >83.02%</td><td   valign=center  nowrap   >85.53%</td><td   valign=center  nowrap   >86.08%</td><td   valign=center  nowrap   >90.43%</td><td   valign=center  nowrap   >88.20%</td><td   valign=center  nowrap   >87.00%</td></tr><tr  ><td  valign=center  nowrap   >LRAF-CNN</td><td   valign=center  nowrap   >90.97%</td><td   valign=center  nowrap   >85.98%</td><td   valign=center  nowrap   >88.40%</td><td   valign=center  nowrap   >88.21%</td><td   valign=center  nowrap   >92.47%</td><td   valign=center  nowrap   >90.29%</td><td   valign=center  nowrap   >89.43%</td></tr></table>					
​													

模型对比实验在COAE2015上的查准率、召回率、F1值、准确率：

<table style="width:100%;"><tr  ><td   valign=center  nowrap  rowspan=2   >模型</td><td   valign=center  nowrap  colspan=3   >负面</td><td   valign=center  nowrap  colspan=3   >中性</td><td   valign=center  nowrap  colspan=3   >正面</td><td   valign=center  nowrap  rowspan=2   >准确率</td></tr><tr  ><td   valign=center  nowrap   >查准率</td><td   valign=center  nowrap   >召回率</td><td   valign=center  nowrap   >F1<font face="华文中宋" >值</font></td><td   valign=center  nowrap   >查准率</td><td   valign=center  nowrap   >召回率</td><td   valign=center  nowrap   >F1<font face="华文中宋" >值</font></td><td   valign=center  nowrap   >查准率</td><td   valign=center  nowrap   >召回率</td><td   valign=center  nowrap   >F1<font face="华文中宋" >值</font></td></tr><tr  ><td   valign=center  nowrap   >CNN</td><td   valign=center  nowrap   >83.02%</td><td   valign=center  nowrap   >80.13%</td><td   valign=center  nowrap   >81.55%</td><td   valign=center  nowrap   >56.00%</td><td   valign=center  nowrap   >23.46%</td><td   valign=center  nowrap   >33.07%</td><td   valign=center  nowrap   >85.64%</td><td   valign=center  nowrap   >94.32%</td><td   valign=center  nowrap   >89.77%</td><td   valign=center  nowrap   >83.78%</td></tr><tr  ><td   valign=center  nowrap   >LAF-CNN</td><td   valign=center  nowrap   >80.19%</td><td   valign=center  nowrap   >88.67%</td><td   valign=center  nowrap   >84.22%</td><td   valign=center  nowrap   >55.91%</td><td   valign=center  nowrap   >30.95%</td><td   valign=center  nowrap   >39.85%</td><td   valign=center  nowrap   >90.83%</td><td   valign=center  nowrap   >91.29%</td><td   valign=center  nowrap   >91.06%</td><td   valign=center  nowrap   >85.41%</td></tr><tr  ><td   valign=center  nowrap   >RAF-CNN</td><td   valign=center  nowrap   >76.29%</td><td   valign=center  nowrap   >88.70%</td><td   valign=center  nowrap   >82.03%</td><td   valign=center  nowrap   >64.00%</td><td   valign=center  nowrap   >9.82%</td><td   valign=center  nowrap   >17.02%</td><td   valign=center  nowrap   >89.71%</td><td   valign=center  nowrap   >92.75%</td><td   valign=center  nowrap   >91.20%</td><td   valign=center  nowrap   >84.87%</td></tr><tr  ><td   valign=center  nowrap   >LRAF-CNN</td><td   valign=center  nowrap   >84.52%</td><td   valign=center  nowrap   >83.72%</td><td   valign=center  nowrap   >84.12%</td><td   valign=center  nowrap   >50.00%</td><td   valign=center  nowrap   >21.23%</td><td   valign=center  nowrap   >29.81%</td><td   valign=center  nowrap   >87.80%</td><td   valign=center  nowrap   >94.23%</td><td   valign=center  nowrap   >90.90%</td><td   valign=center  nowrap   >85.61%</td></tr></table>


## 4、重要说明

本项目中的词向量vectors_300.bin是基于1000W条微博数据训练而成，因此词向量的泛华能力有限，为了覆盖更多的词汇，建议去收集更多领域的语料，然后自己使用word2vec重新训练词向量（当然，这其中还需要经过数据清洗、分词等步骤）。另外，模型的训练语料和测试语料均不足（COAE2014包含4000条、COAE2015包含133201条），无法测试出模型的对于其他特定领域语料的测试效果如何，建议开发者自行测试。



## 5、问题和建议

如果有什么问题、建议、BUG都可以在这个[Issue](https://github.com/9038/ok-emotion-analyzer/issues/1)和我讨论

或者也可以在gitter上交流：[![Gitter](https://badges.gitter.im/superman-projects/ok-emotion-analyzer.svg)](https://gitter.im/superman-projects/ok-emotion-analyzer?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

当然也您可以加入新建的QQ群与我讨论：568079625