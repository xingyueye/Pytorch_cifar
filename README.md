# Pytorch_cifar
Prepare for SRCX AI Contest
------
cifar-10作为分类的基础数据集，已经接近探索极限，目前Gpipe, EfficientNet-b7等网络，已经获得了接近99%的验证精度\
事实上，很多较大的网络，都能很轻易地将训练精度达到100%\
因此，以cifar-10作为AI contest的目标数据集，吐槽之余，应该确立一个有意义的目标：\
`以某个精度较高的基础网络作为baseline，试验各种去过拟合或提高泛化性的算法，最大化压缩验证精度与训练精度的gap`
