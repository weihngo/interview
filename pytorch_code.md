# Optimizer

1、初始化__init__()

    def __init__(self, params, defaults):
        torch._C._log_api_usage_once("python.optimizer")
        self.defaults = defaults
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        # 省略类型检查
        for param_group in param_groups:
            self.add_param_group(param_group)

self.param_groups：保存所有参数信息，以及优化器配置参数。
self.state：保存优化器信息以及参数各网络层索引（索引对应的可在param_groups找到具体参数值）。也可自行在改数据结构中拷贝网络信息。

    def communicate(tensors, communication_op, attention=False):
        flat_tensor = flatten_tensors(tensors)
        communication_op(tensor=flat_tensor, async_op=True)
        if attention:
            return tensors/flat_tensor
        for f, t in zip(unflatten_tensors(flat_tensor, tensors), tensors):
            with torch.no_grad():
                t.set_(f)
                
    communicate(param_list, dist.all_reduce)

通过all_reduce通信进行参数同步

# 问题

1、nn.functional中的函数和nn.Module主要区别：
- nn.Module实现的layers是一个特殊的类，都是有class layer(nn.Module)定义，会自动提取可学习的参数
- nn.functional中的函数更像是纯函数，由def function(input)定义
- 也就是说如果模型有可学习的参数，最好用nn.Module否则使用哪个都可以，二者在性能上没多大差异，
- 对于卷积，全连接等具有可学习参数的网络建议使用nn.Module
- 激活函数（ReLU,sigmoid,tanh），池化等可以使用functional替代。对于不具有可学习参数的层，将他们用函数代替，这样可以不用放在构造函数__init__中。

2、数据相关的dataset和dataloader
- Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
- DataLoader是一个比较重要的类，它为我们提供的常用操作有：batch_size(每个batch的大小), shuffle(是否进行shuffle操作), num_workers(加载数据的时候使用几个子进程)
- Sampler:可以看到初始化参数里有两种sampler：sampler和batch_sampler，都默认为None。前者的作用是生成一系列的index，而batch_sampler则是将sampler生成的indices打包分组，得到一个又一个batch的index。

3、torchvision相关类

- torchvision.datasets：用来进行数据加载的，PyTorch团队在这个包中帮我们提前处理好了很多很多图片数据集。

- torchvision.models：为我们提供了已经训练好的模型，让我们可以加载之后，直接使用。

- torchvision.transforms：提供了一般的 图像转换操作类。（维度转换、旋转、缩放）

- torchvision.utils：

4、分布式训练

- pytorch多卡训练的原理：
  - 将模型加载到一个指定的主GPU上，然后将模型浅拷贝到其它的从GPU上；
  - 将总的batch数据等分到不同的GPU上（坑：需要先将数据加载到主GPU上）；
  - 每个GPU根据自己分配到的数据进行forward计算得到loss，并通过backward得到权重梯度；
  - 主GPU将所有从GPU得到的梯度进行合并并用于更新模型的参数。

- pytorch中gather和scatter_
  - gather（聚合操作）
    - 函数原型：torch.gather(input, dim, index, out=None)；
    - 函数功能：对于out指定位置上的值，去寻找input里面对应的索引位置，根据是index；

    - scatter_（分散操作）
      - 函数原型：Tensor.scatter_(dim, index, src)
      - 函数功能：src（或者说input）指定位置上的值，去分配给output对应索引位置，根据是index；

1. pytorch中torch.Tensor()和torch.tensor()的相同点和区别
  - 相同点：Tensor和tensor都能用于生成新的张量
  - 不同点：
    - torch.Tensor()是python类，是torch.FloatTensor()的别名，使用torch.Tensor()会调用Tensor类的构造函数，生成float类型的张量；
    - torch.tensor()仅仅是python的函数，函数原型是torch.tensor(data, dtype=None, device=None, requires_grad=False)，其中data可以是scalar，list，tuple，numpy array等等。
    - torch.tensor会从data中的数据部分进行拷贝（而不是引用），根据原始数据类型生成相应的 torch.LongTensor、torch.FloatTensor和torch.DoubleTensor。

6. pytorch中Variable的理解
  torch.autograd.Variable是Autograd的核心类，它封装了Tensor，并整合了反向传播的相关实现。Variable包含了三个属性：
  - data：存储了Tensor本体的数据；
  - grad：保存了data的梯度，其本身也是个Variable，shape与data相同；
  - grad_fn：指向Function对象，用于反向传播的梯度计算。但是在pytorch0.4之后，将Variable与Tensor整合到了一起，声明torch.tensor也包含这三个属性。

7. pytorch中backward()的理解
  https://blog.csdn.net/sinat_28731575/article/details/90342082

8. tensorflow中variable和get_variable的区别
  - variable是用来创建变量的，当两个变量的名字在同一作用域内相同时，tensorflow会自动将第二个定义的variable的名字加上"_1"，则会生成新的name，如果使用在name_scope内，则会在name的前面加上name_scope名字的前缀；
  - get_variable的功能是用来进行变量共享的，当变量不存在时，则会自动创建该变量。如果存在时，需要设置reuse=True来获取该变量，实现变量共享。需要注意的是，get_variable在variable_scope内使用才会给name加上前缀，在name_scope中使用并不会。

9. tensorflow中节点和边代表的什么
  - 节点代表着多种功能，比如说输入，变量初始化，运算，控制，输出等；
  - 边代表输入与输出之间的关系，即数据流动的方向。

10. pytorch中train和eval有什么不同
  - model.train()——训练时候启用：启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为True
  - model.eval()——验证和测试时候启用：不启用 BatchNormalization 和 Dropout，将BatchNormalization和Dropout置为False
  train模式会计算梯度，eval模式不会计算梯度。





