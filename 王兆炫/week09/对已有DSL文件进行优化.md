在Dify中排布设计好的workflow,可以导出为yaml/yml文件,称为导出的DSL文件

便于我们移植到新的Dify环境中

### 为Docker配置镜像

因为下方涉及到安装插件,如果不配置镜像,下载速度将非常缓慢,所以这一步也是必要的

在docker中打开设置-Docker Engine

将json格式**添加**如下内容即可
```bash
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://huecker.io",
    "https://dockerhub.timeweb.cloud",
    "https://noohrtwo.mirror.aliyuncs.com"
  ]
```


### 配置大模型API key

初次建立Dify时,需要新增自己的API-key,否则无法访问到对应的大模型

在设置-模型供应商中安装插件即可,安装后添加申请到的api-key

### 完善已有DSL项目

此处要求为 -> 对原始dify中体脂建议应用进行改进，增加一个用户年龄的输入，让应用给出的建议更加完整

我们只需在输入变量,LLM传参,模板转换中均增加一个age变量,让LLM重新整合输出即可

注意这个细节 -> 不同年龄体脂率计算公式是不同的,所以我们需要在计算公式中修改公式,增加年龄的计算

公式如下:
<img width="557" height="166" alt="image" src="https://github.com/user-attachments/assets/3f909627-3b7a-4de5-89bc-e18359442b99" />

修改完毕即可得到新的workflow

<img width="2226" height="1246" alt="image" src="https://github.com/user-attachments/assets/f74a659c-d68e-4730-b3da-c84bb2dd0f73" />

