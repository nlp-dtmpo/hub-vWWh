因为 Dify 不是一个单体桌面软件，而是一个需要同时运行前端、后端以及多种基础服务的开源平台；

其面向的是自托管部署场景,所以不同于平常的软件,可以直接一键安装,我们在这里需要使用
Docker Compose部署Dify

### 安装Docker

在 [Docker官网](https://www.docker.com/products/docker-desktop/) 下载对应版本的Docker

注意安装好之后要重启电脑,提前做好备份

### 更新WSL

打开docker后,docker会自动弹出需要执行的命令行,直接执行安装即可

安装完毕重新打开即可

### 部署Dify

参照[Dify部署指南](https://docs.dify.ai/zh/self-host/quick-start/docker-compose)

即可安装Dify

### 打开Dify

使用如下网址即可访问到Dify
```bash
http://localhost/install
```
