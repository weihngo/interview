# 项目总结
## 一、阿里天池服务调度比赛
### 项目介绍：
在项目中我主要负责数据特征处理和调度算法设计。
在数据特征处理中，针对大规模的数据特征，主要采用主成分分析法对任务特征进行数据降维，避免在后期匹配时出现过拟合现象，并采用奇异值分解对各服务专员与任务间进行专属度分析，由此可在迭代过程中使得服务专员更趋向于处理其专属度高的任务，专属度同时考虑了任务可执行性和任务处理效率。
在调度算法中，针对大规模的匹配工作，主要使用元启发式算法进行最优调度策略的生成，在对比各类元启发式算法后决定使用粒子群算法，因在各算法中都存在迭代后期多粒子移动步调一致性问题，而我在使用粒子群算法中发现动态调整粒子全局最优速度和局部最优速度的比重可以避免移动步调一致性问题，并加入随机速度增加粒子变异概率。
### Q-A：
Q：调度问题的抽象？
A：加上时间限制的任务匹配问题，在任务开始时间的限制下，根据各任务匹配各专员的损失找出使得损失最小的最优解。

Q：能不能用机器学习的方法求解？
A：可以用强化学习解决调度问题，拟定好奖励函数与惩罚函数，奖励函数应为超时允许范围内的运行效率收益，惩罚函数为超时后的时间损失，并在迭代中加入正则化防止过拟合。在图算法中可以把问题抽象成多路并行遍历结点问题，每一路代表一个服务专员，图上每个节点代表每个任务，在最高收益前提下遍历图内的所有点。

## 二、肾小球图像分割比赛
### 项目介绍：
在项目中我主要负责用pytorch实现图像语义分割算法，针对模型选择，我在对比并尝试过多种模型之后发现，由于Unet中的下采样次数较少，对图像中的小目标得以保留，最适合用于医学影像的语义分割。为提高模型精度，我们后期还采用了模型融合方法，在Unet结构下采样过程中加入了deeplabv3+中的空洞卷积和FPN网络结构，将其结果与Unet结果进行链接，并参考SENet结构加入注意力机制，最后对模型精度有了大幅度的提升。
### Q-A：
Q：对目标识别模型有一定理解吗？
A：有，主要了解SSD和YOLO等anchor-based模型和centerNet等anchor-free模型。

anchor-free发展：DenseBox、YOLO、CornerNet、ExtremeNet、FSAF、FCOS、FoveaBox。

## 三、基于异构边缘推理设备的目标识别系统
### 项目介绍
本项目是由20个异构推理节点组成的推理后端，其中包括：agx、TX2、瑞芯微RK3399、寒武纪智能计算卡等设备，并将YOLO、SSD等剪枝量化后的目标识别模型部署在各设备上，由服务端统一调度。在服务端中根据神经网络结构来分析其在各硬件上的性能表征，再通过改进后的粒子群做任务调度。本人主要负责异构平台的系统及运行库适配工作，针对异构平台的神经网络模型移植、编写推理终端中的目标识别程序，并与服务端进行socket通信，以及调度算法设计。

移植模型有：MobileNetV2-SSD、VGG16-SSD、普通yolov3模型和遥感目标识别的yolov3模型。遥感目标识别采用darknet的yolov3通过对大尺寸遥感图像切割进行训练，并采用一些数据增强方法（翻转、色彩调整、裁剪、随机擦除等）。

- 模型移植
  - nvidia设备可以直接运行darknet模型，并根据darknet转tensorrt进行推理加速。
  - rk3399中将darknet训练所得模型通过fp16量化成rknn模型，并根据测试集提供的示例图片进行参数微调
  - 寒武纪模型移植中主要针对darknet转为寒武纪所支持的cambricon模型，该移植过程需要先进行darknet转caffe再转寒武纪离线推理模型。
    - 由于寒武纪板卡中仅支持网络的特征图的推理计算，没有yolov3模型中最后检测推理层和非极大值抑制算子，所以需要对算子进行重写，并在cpu上执行计算。
    - 为提升推理速度，我们对图像前处理、推理、后处理三个过程做了数据流水，通过多线程实现，并采用加锁后的管道通信维护数据读写一致。最后使得在cpu资源占用最少的前提下达到计算速度最优。
    - 为使得模型更小更快，网络训练过程中采用剪枝方式进行模型压缩，通过对特征图各通道图层的重要性进行通道剪枝，大幅减少网络参数量和计算量，最后将300+MB的yolov3模型压缩至30+MB，并由于模型剪枝解决了网络过拟合问题，最后在测试集的精度表现更优

- 模型剪枝
  - 神经元重要性衡量方法：
    - **思路一：** 按参数的绝对值大小评估重要性，贪心的选择部分删去。在训练时针对性的在损失函数中加入L1正则化，使得权重稀疏化。此外，重要性还可以作用在归一化层和激活函数上：
      - 在BN层加入channel-wise scaling factor 并对之加L1 regularizer使之稀疏，然后裁剪scaling factor值小的部分对应权重；
      - 像Relu这样的激活函数会倾向产生稀疏的activation；而权重相对而言不太容易是稀疏的（当前，如前所说，我们可以通过regularizer这种外力使它变得稀疏）。
    - **思路二：** 考虑参数裁剪对loss的影响。《Pruning Convolutional Neural Networks for Resource Efficient Transfer Learning》采用的是目标函数相对于activation的展开式中一阶项的绝对值作为pruning的criteria，这样就避免了二阶项（即Hessian矩阵）的计算。2018年论文《SNIP: Single-shot Network Pruning based on Connection Sensitivity》将归一化的目标函数相对于参数的导数绝对值作为重要性的衡量指标。
    - **思路三：** 考虑对特征输出的可重建性的影响，即最小化裁剪后网络对于特征输出的重建误差。如果对当前层进行裁剪，然后如果它对后面输出还没啥影响，那说明裁掉的是不太重要的信息。
      - 通过最小化特征重建误差（Feature reconstruction error）来确定哪些channel需要裁剪，裁剪方法包括：贪心法，LASSO regression。
      - NISP（Neuron importance score propagation）算法通过最小化分类网络倒数第二层的重建误差，并将重要性信息反向传播到前面以决定哪些channel需要裁剪。
      - DCP（Discrimination-aware channel pruning）方法一方面在中间层添加额外的discrimination-aware loss（用以强化中间层的判别能力），另一方面也考虑特征重建误差的loss，综合两方面loss对于参数的梯度信息，决定哪些为需要被裁剪的channel。
    - **思路四：** 基于其它的准则对权重进行重要性排序。FPGM（Filter Pruning via Geometric Median）方法，基本思想是基于geometric median来去除冗余的参数。我们知道贪心算法的缺点就是只能找到局部最优解，因为它忽略了参数间的相互关系。那自然肯定会有一些方法会尝试考虑参数间的相互关系，试图找导全局更优解。
  - 剪枝方法：
    - **离散空间下的搜索：** 如2015年的论文《Structured Pruning of Deep Convolutional Neural Networks》基于genetic algorithm与particle filter来进行网络的pruning。2017年的论文《N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning》尝试将网络的压缩分成两个阶段-layer removal和layer shrinkage，并利用强化学习（Reinforcement learning）分别得到两个阶段的策略。
    - **规划问题：** 如比较新的2019年论文《Collaborative Channel Pruning for Deep Networks》提出CCP（Collaborative channel pruning）方法，它考虑了channel间的依赖关系 ，将channel选取问题形式化为约束下的二次规划问题，再用SQP（Sequential quadratic programming）求解。
    - **Bayesian方法：** 如2017年论文《Variational Dropout Sparsifies Deep Neural Networks》提出了sparse variational droput。它对variational droput进行扩展使之可以对dropout rate进行调节，最终得到稀疏解从而起到裁剪模型的效果。
    - **基于梯度的方法：** 回顾上面问题定义中的数学最优化问题，其最恶心的地方在于regularizer中那个L0-norm，使目标不可微，从而无法用基于梯度的方法来求解。如2017年的论文《Learning Sparse Neural Networks through L0 Regularization》的思路是用一个连续的分布结合 hard-sigmoid recification去近似它，从而使目标函数平滑，这样便可以用基于梯度的方法求解。
    - **基于聚类的方法：** 一般地，对于压缩问题有一种方法就是采用聚类。如将图片中的颜色进行聚类，就可以减小其编码长度。类似地，在模型压缩中也可以用聚类的思想。如2018年的论文《SCSP: Spectral Clustering Filter Pruning with Soft Self-adaption Manners》和《Exploring Linear Relationship in Feature Map Subspace for ConvNets Compression》分别用谱聚类和子空间聚类发掘filter和feature map中的相关信息，从而对参数进行简化压缩。
