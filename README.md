# Classifying Dogs and Cats ussing CNN (MM803 Project- Image and Video Processing)
![alt tag](https://github.com/shrobon/Classifying-Dogs-and-Cats-using-CNN/blob/master/banner.png)

The aim of this project is to use Deep Learning as a tool to correctly classify images of cats and dogs,using a subset of the Asirra dataset. To foster a good understanding, and appreciate some Deep Learning techniques and models, the project report has been drafted such that, every new experiment leads to an incremental growth in performance, compared to the previous experiment.
**MM803_Project_Report.pdf file contains the full project report.**

In this project, the different techniques like data augmentation, batch normalization, and weight
initialization were studied and their results were compared. I was able to get a classification accuracy of 90.18%, without the use of an external dataset. This accuracy can further be improved by making just slight changes to the existing model by fine tuning the hyperparameters even more. Due to hardware constraints, I had to limit myself to at most of 200 epochs. This leaves a big scope for future work to be done using different activation functions like pRelu and leakyRelu, different models, and benchmarking their performance.

## Watch this youtube video
[![Demo Video](https://github.com/shrobon/Classifying-Dogs-and-Cats-using-CNN/blob/master/Dogs_cats.png)](https://www.youtube.com/watch?v=SfeCFWZIr3Q "Watch this Demo Video")

## The following softwares are required to run this project
1. Python 2.7
2. Keras 1.1.2
3. Theano 0.9.0.dev4

## Labeled Images for training
```
https://drive.google.com/open?id=0B5L6sQPsKnRfV2k0Q3Z0VThPQTA
```

## After training, test the model using these images 
```
https://drive.google.com/open?id=0B5L6sQPsKnRfZkFpalVtTmhUZXc
```
## To start training, type the following in your terminal
```
python demo1.py -train train_basic
```

## After Training, use type the following your terminal, to Test with Images
```
python exp1_Test_batch.py -image test_classification
```
## Training Loss VS Validation Loss
![alt tag](https://github.com/shrobon/Classifying-Dogs-and-Cats-using-CNN/blob/master/figure_1-3.png)


## Experiements were performed using the following hardware
1. Processor: Intel Core i7-6700HQ(2.6Ghz)
2. RAM: 16GB DDR4
3. GPU: Nvidia GTX 1060 (6GB)
 
## License
MIT License

Copyright (c) 2016 Shrobon Biswas

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
