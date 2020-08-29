## GradCAM-and-GradCAM-
Based on pytoch, gradcam and gradcam + + are encapsulated into easy-to-use API, and some interesting tests are done with pre trained vgg16, alexnet, densenet 121, mobilenet, resnet18, squeezene and I made a very detailed comment on the code. Interested friends can give a star, thank you.

## How to run
Open the notebook and run it directly.

## Interesting experiments
The following are the predicted results of vgg16, alexnet, densenet 121, mobilenet, resnet18, squeezene model and the heat map of gradcam + +. The prediction of each model is：  
vgg16 : bullet train, bullet   
alexnet : bullet train, bullet   
densenet121 : bullet train, bullet   
mobilenet_v2 : bullet train, bullet   
resnet18 : bullet train, bullet   
squeezenet : bullet train, bullet  
![test1_picture](https://github.com/Greak-1124/GradCAM-and-GradCAM-/blob/master/Outputs/Test.JPEG)

However, I reduced the bullet train to cover the plane graph, and the prediction results of each model became different. By observing their heat maps, we can roughly know why their predictions are different. Through the heat map, we can see that different parts of each model focus on, which will lead to different final prediction results.The prediction of each model is：  
vgg16 : mouse, computer mouse   
alexnet : pencil sharpener   
densenet121 : space shuttle   
mobilenet_v2 : bullet train, bullet   
resnet18 : wing   
squeezenet : projectile, missile  
![test2_picture](https://github.com/Greak-1124/GradCAM-and-GradCAM-/blob/master/Outputs/Test1.JPEG)

## Summary
We can use the heat map to analyze the prediction basis of a model. Through the heat map, we can roughly know why it predicts this object into this category, so that developers can improve the model or enhance the data to improve the robustness of the model.  

## References
[1]https://github.com/1Konny/gradcam_plus_plus-pytorch  
[2] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, Selvaraju et al, ICCV, 2017  
[3] Grad-CAM++: Generalized Gradient-based Visual Explanations for Deep Convolutional Networks, Chattopadhyay et al, WACV, 2018  
