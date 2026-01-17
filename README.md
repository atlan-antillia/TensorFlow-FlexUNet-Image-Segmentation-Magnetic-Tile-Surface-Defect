<h2>TensorFlow-FlexUNet-Image-Segmentation-Magnetic-Tile-Surface-Defect  (2026/01/17)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>Magnetic-Tile-Surface-Defect</b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> 
(TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and an <a href="https://drive.google.com/file/d/1EORUyF2idLaIOvpI85KfWKjUmpdGJ3m2/view?usp=sharing"><b>Augmented-Magnetic-Tile-ImageMask-Dataset.zip</b></a> dataset with colorized masks, which was derived by us from <br><br>
<a href="https://github.com/abin24/Magnetic-tile-defect-datasets.">
<b>Magnetic-tile-defect-datasets</b>.</a>
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of images and masks of the original <b>Magnetic-tile-defect-datasets</b> ,
<!-- which contains 605 GeoTIFF images and their corresponding TIF masks  respectively,
-->
we generated  the  Augmented dataset by using an offline augmentation tool 
<a href="./generator/ImageMaskDatasetGenerator.py">
ImageMaskDatasetGenerator.py</a>.
<br><br> 
<hr>
<b>Actual Image Segmentation for Magnetic-Tile-Surface-Defect Images </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {Blowhole:red, Break:green, Crack:blue, Fray:yellow,  Uneven:mazenda}</b><br>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/Blowhole_104.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/Blowhole_104.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/Blowhole_104.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/Fray_24.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/Fray_24.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/Fray_24.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/Break_4.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/Break_4.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/Break_4.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://github.com/abin24/Magnetic-tile-defect-datasets.">
Magnetic-tile-defect-datasets.</a>
<br><br>
Please refer to: <a href="https://link.springer.com/article/10.1007/s00371-018-1588-5">Surface defect saliency of magnetic tile</a>
<b></b></a>
<br><br>
The following explanation was taken from <a href="https://github.com/abin24/Magnetic-tile-defect-datasets.">
Magnetic-tile-defect-datasets.</a><br><br>
This is the datasets of the paper "Saliency of magnetic tile surface defects".
 The images of 6 common magnetic tile defects were collected, and their pixel level ground-truth were labeled.<br>
<ul>
<li>Blowhole</li>
<li>Break</li>
<li>Crack</li>
<li>Fray</li>
<li>Free</li>
<li>Uneven</li>
</ul>
<br>
<b>License</b><br>
Please refer to <a href="https://github.com/abin24/Surface-Inspection-defect-detection-dataset">
Surface-Inspection-defect-detection-dataset.
</a><br>
<b>The image datasets are only for academic research, no commercial purposes are allowed. If you use any datasets, 
please cite the paper of the corresponding provider</b>
<br>
<br>
<h3>
2 Magnetic-Tile ImageMask Dataset
</h3>
 If you would like to train this Magnetic-Tile Segmentation model by yourself,
please download the master  dataset from
<a href="https://drive.google.com/file/d/1EORUyF2idLaIOvpI85KfWKjUmpdGJ3m2/view?usp=sharing"><b>Augmented-Magnetic-Tile-ImageMask-Dataset.zip</b></a> 
<br>
, expand the downloaded, and  put it under <b>./dataset</b> folder to be:
<pre>
./dataset
└─Magnetic-Tile
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Magnetic-Tile Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Magnetic-Tile/Magnetic-Tile_Statistics.png" width="512" height="auto"><br>
<br>
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Magnetic-Tile TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Magnetic-Tile/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Magnetic-Tile and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False
num_classes    = 6
base_filters   = 16
base_kernels  = (11,11)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Magnetic-Tile 1+5 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Magnetic-Tile 1+5
rgb_map = {(0,0,0):0, (255, 0, 0):1, (0,255,0):2, (0,0,255):3,(255,255,0):4,(255,0,255):5,}

</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (17,18,19)</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (35,36,37)</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 37 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/train_console_output_at_epoch37.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Magnetic-Tile/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Magnetic-Tile/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Magnetic-Tile</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Magnetic-Tile.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/evaluate_console_output_at_epoch37.png" width="880" height="auto">
<br><br>Image-Segmentation-Magnetic-Tile

<a href="./projects/TensorFlowFlexUNet/Magnetic-Tile/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Magnetic-Tile/test was not low, but dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0346
dice_coef_multiclass,0.98
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Magnetic-Tile</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Magnetic-Tile.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for  Steel-Rail-Surface-Defect Images </b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks.
<br>
<b>rgb_map = {Blowhole:red, Break:green, Crack:blue, Fray:yellow,  Uneven:mazenda}</b><br>
<br>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/Blowhole_104.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/Blowhole_104.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/Blowhole_104.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/hflipped_Break_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/hflipped_Break_1.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/hflipped_Break_1.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/Crack_11.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/Crack_11.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/Crack_11.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/Fray_24.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/Fray_24.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/Fray_24.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/hflipped_Uneven_8.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/hflipped_Uneven_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/hflipped_Uneven_8.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/images/vflipped_Fray_28.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test/masks/vflipped_Fray_28.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Magnetic-Tile/mini_test_output/vflipped_Fray_28.png" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
<h3>
References
</h3>
<b>1. Segmentation_Method_of_Magnetic_Tile_Surface_Defects_Based_on_Deep_Learning</b><br>
Y. An, Y.N. Lu, T.R. Wu<br>
<a href="https://www.researchgate.net/publication/359331481_Segmentation_Method_of_Magnetic_Tile_Surface_Defects_Based_on_Deep_Learning">
https://www.researchgate.net/publication/359331481_Segmentation_Method_of_Magnetic_Tile_Surface_Defects_Based_on_Deep_Learning</a>
<br>
<br>
<b>2. A novel dual-student reverse knowledge distillation method for magnetic tile defect detection</b><br>
Jiyan Tang, Ao Zhang & Weian Liu <br>
<a href="https://www.nature.com/articles/s41598-025-12339-2">
https://www.nature.com/articles/s41598-025-12339-2
</a>
<br>
<br>
<b>3. Surface Defect Detection of Magnetic Tiles Based on YOLOv8-AHF</b><br>
Cheng Ma, Yurong Pan andJunfu Chen<br>
<a href="https://www.mdpi.com/2079-9292/14/14/2857">
https://www.mdpi.com/2079-9292/14/14/2857
</a>
<br>
<br>
<b>4. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
