# PHG
Parallel Horizon Gene
# 配置方法：
## 一、配置MPI框架
1.  http://www.mpich.org/downloads/
2. 找到Microsoft Windows-[http]-点击
3. 点击下载MS-MPI v10.x.x
4. 全选下载，各自安装
5. 打开C++项目的属性：
- (1)C/C++—预处理器—编辑—MPICH_SKIP_MPICXX
- (2)C/C++—代码生成—运行库—多线程调试（/MTD）
- (3)链接器—输入—附加依赖项—msmpi.lib
- (4)VC++目录—包含目录—$(MPI安装位置)\Microsoft SDKs\MPI\Include
- (5)VC++目录→库目录—$(MPI安装位置)\Microsoft SDKs\MPI\Lib\x64
- (6)调试—命令—mpiexec.exe文件的位置
- (7)调试—命令参数—-n N $(TargetPath)；其中N为设定的线程数
## 二、配置C++ boost库
1. https://www.boost.org/users/download/
2. 解压并打开文件夹
3. 运行bootstrap.bat
4. powershell/cmd进入对应路径，运行.\b2.exe --address-model=64 --with-mpi runtime-link=static
5. VC++目录—包含目录—$(Boost安装位置)
6. VC++目录→库目录—$(Boost安装位置)\stage\lib
