##Automatic Portrait Segmentation for Image Stylization##

Created by Xiaoyong Shen ( goodshenxy@gmail.com ) at CUHK. 

Please cite the following [paper](http://xiaoyongshen.me/webpage_portrait/papers/portrait_eg16.pdf):

	@article {Autoportraitseg,
	author = {Shen, Xiaoyong and Hertzmann, Aaron and Jia, Jiaya and Paris, Sylvain and Price, Brian and Shechtman, Eli and Sachs, Ian},
	title = {Automatic Portrait Segmentation for Image Stylization},
	journal = {Computer Graphics Forum},
	volume = {35},
	number = {2},
	pages = {93--102},
	year = {2016},
}

### Structures of the Files

The code includes four folders:

1. *Caffe-portraitseg*: implementations of our Caffe.
2. *data*: scripts for downloading data.
    - *data/images_mask*: all our labeled portrait masks.
    - *data/images_tracker*: face tracker points of all portraits.
    - *data/trainlist.mat*: the image IDs of training set.
    - *data/testlist.mat*: the image IDs of test set.
    - *data/alldata_urls.txt*: the URLs of all images in Flickr.
    - *data/crop.txt*: the cropping parameters of each downloaded portrait.
3. *training*: scripts and models for training.
    - *training/FCN8s_models*: the initial models from FCN.
    - *training/model_files*: network and solver files for our models.
4. *testing*: scripts and models for testing.
    - *testing/our_models*: our pre-trained models.


### Download Data ###
In Matlab, the data can be downloaded by

    >>cd ./data
    >>data_download.m
The images are saved in folder *images\_data\_crop* and the corresponding masks are in *images\_mask*. 

There are also **122 images** cannot be downloaded via our provided URLs. Please send Email to *goodshenxy@gmail.com* to obtain it. Then, copy all you obtained images into the folder *./data/images_data_crop* after you run the **data\_download.m**.

### Install Caffe ###

1. Install prerequisites for Caffe ( [http://caffe.berkeleyvision.org/installation.html#prequequisites](http://caffe.berkeleyvision.org/installation.html#prequequisites) ). Our provided Caffe also need to install the [Matio](https://sourceforge.net/p/matio/matio/ci/matio_1_5_8/tree/).
2. Compile the Caffe provided in folder *Caffe-portraitseg*.

### Prepare Files for Training and Testing ###
In Matlab, run

    >>cd ./data
    >>generate_data_file.m

Two folders and two text files will be generated. They are

- *portraitFCN\_data*: data for training and testing PortraitFCN model.
- *portraitFCN+\_data*: data for the PortraitFCN+ model.
- *portraitFCN\_datalist.txt*: data list to train the PortraitFCN model.
- *portraitFCN+\_datalist.txt*: data list to train the PortraitFCN+ model. 

### Training ###
To train the PortraitFCN model,

    $ cd ./training
    $ python train_portraitFCN.py

and

    $ python train_portraitFCN+.py
for PortraitFCN+ model.


### Testing ###

Our pre-trained models are stored in *./testing/our_models*. In Matlab,

    >>cd ./testing
    >>demo_portraitFCN.m
    >>demo_portraitFCNplus.m
are the scripts to get the segmentation of one input image.

To evaluate the models,

    >>test_portraitFCN.m
output the accuracy of PortraitFCN model in our testing dataset, and

    >>test_portraitFCNplus.m
achieves the accuracy of PortraitFCN+ model.


### License

This software and data is released only for personal and research use. For a commercial license please contact the authors.

### Ackowledgement ###

This code is based on Caffe. Thanks to the contributors of Caffe. Thanks @shelhamer and @longjon for providing fundamental implementations that enable fully convolutional training/testing in Caffe.

	@misc{Jia13caffe,
      Author = {Yangqing Jia},
      Title = { {Caffe}: An Open Source Convolutional Architecture for Fast Feature Embedding},
      Year  = {2013},
      Howpublished = {\url{http://caffe.berkeleyvision.org/}}
    }

If you encounter any issue when using our code or model, please let me know.


