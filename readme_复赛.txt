队伍：megemini
成绩：0.770 （2018-12-05） 
文件：1544004574135_s2_01_final_v4_b_raw.zip

（1.简要介绍每个文件的作用及自己实现的代码主要在哪些文件）
1. 目录结构：
|-run.ipynb # 代码，只有一个文件，直接运行程序即可。程序会先训练模型，然后预测并生成结果。
|-readme.txt # 本说明文档
|-Demo 
  |-DataSets # 数据文件，内部结构保持不变
  |-Models 
    |-s2_01 # 存放模型文件，以及训练历史文件。复现时请新建此文件夹！！！
  |-result
    |-s2_01_final_v4_b_raw # 存放结果文件。复现时请新建此文件夹！！！
    
（2.注明哪些部分使用到开源工具）
2. 程序主要用到开源工具
（1）jieba、gensim，预训练词向量
（2）keras，建模

（3.复现时需要在一个新的dsw环境下额外安装哪些包）
3. 文件后面使用pip freeze导出了所有安装包，复现时相应安装即可。

（4.运行时长，分别注明训练时长和预测时长）
4. 时长：
（1）数据准备：～5min
（2）模型训练：～868（每epoch）×25（epoch数）×2（正反两次）sec = 723.3min = 12hour
（3）模型预测：～36min

（5.生成的提交结果文件所在位置）
5. 目录：./Demo/result/s2_01_final_v4_b_raw

（没有6）

（7.队伍名称 队长简介及联系方式）
7. 队伍
名称：megemini
邮箱：megemini@outlook.com

（8.其他需特殊说明的部分，没有则不写。）
8. 其他事项
（1）由于使用jupyter notebook进行编辑，notebook中会记录每一步运行的结果，如果出入较大则考虑代码运行出错。
（2）由于之前run.ipynb文件是在Demo文件夹内，而且会在Demo中的Models文件夹中产生模型文件，复现如果出现路径错误请联系队长修改。
（3）打包ann目录的时候，请务必过滤掉ipynb的隐藏文件，以防结果出错！！！

9. 解题思路
基本思路为：(1)FastText生成embedding向量 -> (2)Stack-Bi-LSTM（GRU）模型 -> (3)Bi-Sample Training
(1)FastText生成embedding向量
复赛数据为arg_a(arg_1)和arg_b(arg_2)的匹配问题，本程序针对arg_b进行滑窗采样以生成训练数据。
这就可能会产生一个问题，由于arg_b在text中的分布不均匀导致网络词向量不能很好得到训练，而且随着滑窗的缩小，问题可能越来越明显。所以，这里先使用train text训练FastText生成词向量，再导入Stack-Bi-LSTM模型进行训练。
(2)Stack-Bi-LSTM（GRU）模型
Stack是指三个输入：text、arg_a、arg_b分别通过embedding和Bi-LSTM后连接到一起，再一同送入Bi-LSTM、Bi-GRU模型这样的一个模型结构。
其中text为滑窗范围内（此处为320）的文本编码；arg_a为滑窗范围内所有arg_a的索引值；arg_b为此滑窗内需要匹配的某一个arg_b索引值。
模型的输出为，此滑窗内针对此arg_b的arg_a的0/1真值序列。
(3)Bi-Sample Training
Bi-LSTM和Bi-GRU虽然为双向结构，但文本本身会存在左右顺序，所以这里使用正向、反向两份数据进行训练。也就相当于数据增强了一倍。但是，如果将两份数据混在一起训练，可能会彼此产生噪音，所以，在训练时间相同的情况下，正反向分别训练一个模型然后融合预测。

10. pip 环境

absl-py==0.6.1
asn1crypto==0.23.0
astor==0.7.1
backcall==0.1.0
bleach==3.0.2
boto==2.49.0
boto3==1.9.57
botocore==1.12.57
bz2file==0.98
certifi==2018.10.15
cffi==1.11.2
chardet==3.0.4
conda==4.3.31
cryptography==2.1.4
cycler==0.10.0
decorator==4.3.0
defusedxml==0.5.0
docutils==0.14
entrypoints==0.2.3
future==0.16.0
gast==0.2.0
gensim==3.6.0
grpcio==1.16.0
h5py==2.8.0
idna==2.7
ipykernel==5.1.0
ipython==7.1.1
ipython-genutils==0.2.0
jedi==0.13.1
jieba==0.39
Jinja2==2.10
jmespath==0.9.3
jsonschema==2.6.0
jupyter-client==5.2.3
jupyter-core==4.4.0
jupyterlab==0.34.12
jupyterlab-launcher==0.13.1
jupyterlab-prometheus==0.1
Keras==2.2.4
Keras-Applications==1.0.6
Keras-Preprocessing==1.0.5
keras-self-attention==0.31.0
kiwisolver==1.0.1
lightgbm==2.2.2
Markdown==3.0.1
MarkupSafe==1.1.0
matplotlib==3.0.0
mistune==0.8.4
nbconvert==5.4.0
nbformat==4.4.0
notebook==5.7.0
np-utils==0.5.5.2
numpy==1.15.2
pandas==0.23.4
pandocfilters==1.4.2
parso==0.3.1
pexpect==4.6.0
pickleshare==0.7.5
Pillow==5.3.0
prometheus-client==0.4.2
prompt-toolkit==2.0.7
protobuf==3.6.1
ptyprocess==0.6.0
pycosat==0.6.3
pycparser==2.18
Pygments==2.2.0
pyOpenSSL==17.5.0
pyparsing==2.2.2
PySocks==1.6.7
python-dateutil==2.7.5
pytz==2018.5
PyYAML==3.13
pyzmq==17.1.2
requests==2.19.1
ruamel-yaml==0.11.14
s3transfer==0.1.13
scikit-learn==0.20.0
scipy==1.1.0
Send2Trash==1.5.0
simplegeneric==0.8.1
simplejson==3.16.0
six==1.11.0
smart-open==1.7.1
tensorboard==1.11.0
tensorflow-gpu==1.11.0
termcolor==1.1.0
terminado==0.8.1
testpath==0.4.2
torch==0.4.1
torchtext==0.3.1
torchvision==0.2.1
tornado==5.1.1
tqdm==4.28.1
traitlets==4.3.2
urllib3==1.23
wcwidth==0.1.7
webencodings==0.5.1
Werkzeug==0.14.1
xgboost==0.81

