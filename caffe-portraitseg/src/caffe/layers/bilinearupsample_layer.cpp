#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
//#include "caffe/util/rearrange.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace caffe {

template <typename Dtype>
void BilinearupsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  BilinearupsampleParameter bilinear_param = this->layer_param_.bilinearupsample_param();
   if(!bilinear_param.has_num_output_h()){
    num_output_h_=bilinear_param.num_output();
    num_output_w_=bilinear_param.num_output();
  }else{
    num_output_h_=bilinear_param.num_output_h();
    num_output_w_=bilinear_param.num_output_w();
  }
  // Configure output channels and groups. we operate bilinearolution per channel
 // channels_ = bottom[0]->channels();
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  channels_=bottom[0]->channels();
  
}

template <typename Dtype>
void BilinearupsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {

  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_,channels_, num_output_h_, num_output_w_);
  }

}

template <typename Dtype>
void BilinearupsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
      
    Dtype* ptr_bottom_data = bottom[0]->mutable_cpu_data();
    Dtype* ptr_top_data=top[0]->mutable_cpu_data();
      cv::Mat temp_mat(height_,width_,CV_32FC1);
      cv::Mat temp_out(num_output_h_,num_output_w_,CV_32FC1);
      //using bilinear to upsample the feature map
    for (int n = 0; n < num_; ++n) 
    {
      for (int dim =0 ;dim<channels_;++dim){

          for (int h=0;h<height_;++h)
          {
          	for(int w=0;w<width_;++w)
          	{
          		temp_mat.at<float>(h,w) = ptr_bottom_data[((n * channels_+dim) * height_+ h) * width_+ w];
             // LOG(INFO)<<temp_mat.at<float>(h,w) ;
          	}
          }
       if(this->layer_param_.bilinearupsample_param().samplemethod()
        == BilinearupsampleParameter_SampleMethod_NEAREST)
      {cv::resize(temp_mat, temp_out, cv::Size(num_output_h_, num_output_w_),0,0,cv::INTER_NEAREST);
       // LOG(INFO)<<"NEAREST";
      }
    else cv::resize(temp_mat, temp_out, cv::Size(num_output_h_, num_output_w_));
    
       for(int h_out=0; h_out<num_output_h_;++h_out)
       {
    	for(int w_out=0;w_out<num_output_w_;++w_out)
    	{
    		ptr_top_data[((n * channels_+dim) * num_output_h_+ h_out) * num_output_w_+ w_out] = temp_out.at<float>(h_out,w_out);
    	}
       }
    }
  
   }

   //check data 
  }

template <typename Dtype>
void BilinearupsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  // LOG(INFO)<<"BEGIN";
  bool backward_flag = this->layer_param_.bilinearupsample_param().backwardflag();
  if(backward_flag){
  //backward bilinear doen sample 
   Dtype* top_diff = top[0]->mutable_cpu_diff();
   Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
   cv::Mat temp_mat(num_output_h_,num_output_w_,CV_32FC1);
   cv::Mat temp_out(height_,width_,CV_32FC1);
   for (int n = 0; n < num_; ++n) 
    {
      for (int dim =0 ;dim<channels_;++dim){

          for (int h=0;h<num_output_h_;++h)
          {
          	for(int w=0;w<num_output_w_;++w)
          	{
          		temp_mat.at<float>(h,w) = top_diff[((n * channels_+dim) * num_output_h_+ h) * num_output_w_+ w];
             // LOG(INFO)<<top_diff[((n * channels_+dim) * num_output_h_+ h) * num_output_w_+ w];
          	}
          }
      if(this->layer_param_.bilinearupsample_param().samplemethod()
        == BilinearupsampleParameter_SampleMethod_NEAREST)
      {cv::resize(temp_mat, temp_out, cv::Size(height_, width_),0,0,cv::INTER_NEAREST);
      //  LOG(INFO)<<"NESREST";
      }
    else  cv::resize(temp_mat, temp_out, cv::Size(height_, width_));
    
       for(int h_out=0; h_out<height_;++h_out)
       {
    	for(int w_out=0;w_out<width_;++w_out)
    	{
    	      bottom_diff[((n * channels_+dim) * height_+ h_out) * width_+ w_out] = temp_out.at<float>(h_out,w_out);
           
    	}
       }
    }
  
  
}


 // else LOG(INFO)<<"This Layer doesn't need backpropagation!";
}
}


#ifdef CPU_ONLY
STUB_GPU(BilinearupsampleLayer);
#endif

INSTANTIATE_CLASS(BilinearupsampleLayer);
REGISTER_LAYER_CLASS(Bilinearupsample);


}  // namespace caffe
