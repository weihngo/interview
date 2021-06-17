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


