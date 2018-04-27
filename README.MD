# Portrait Segmentation in PyTorch
Based on Automatic Portrait Segmentation for Image Stylization (Shen et al. 2016)

### Notes and Instructions
```FCN.py``` contains a PyTorch implementation of FCN-8s (Long et al. 2015), using the convolutional part of the model architecture of VGG-16 (Simonyan et al. 2014).

Execute using ```python run.py```

```train()``` and ```test()``` methods are in ```trainer.py```

Code is meant for the data from Shen et al. The data is handled through ```DataLoader.py``` for this particular dataset. Modifications to this file might be necessary to adapt to a different dataset.

### References
Long, J., Shelhamer, E. and Darrell, T., 2015. Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).

Shen, X., Hertzmann, A., Jia, J., Paris, S., Price, B., Shechtman, E. and Sachs, I., 2016, May. Automatic portrait segmentation for image stylization. In Computer Graphics Forum (Vol. 35, No. 2, pp. 93-102).

Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
