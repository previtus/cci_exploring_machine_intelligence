# Exploring Machine Intelligence

**Instructor:** Vit Ruzicka

**Email:** v.ruzicka@arts.ac.uk

**Class moodle:** https://moodle.arts.ac.uk/course/view.php?id=38156

### Assignments and notes will be posted weekly on github here:

    https://github.com/previtus/cci_exploring_machine_intelligence

### Video recordings:

* YouTube playlist: [CCI - Exploring Machine Intelligence](https://www.youtube.com/playlist?list=PLCIVpmFkFKQ88lzWtYW2MCwqXofhkzqgt)

### Course Description:

This advanced unit introduces Machine Learning (ML) approaches, concepts and methods through direct examples, practical problem solving and core technical training for creative applications.

The unit explores a specific set of approaches for both interactive and offline Machine Learning using common tools and frameworks such as Tensorflow, Torch, Keras and TFlearn, or equivalent. Fundamental concepts such as classification, clustering and regression are developed through practical problem solving including gesture recognition and tracking, sound and image classification, and text processing. Problems such sequence matching using probabilistic and stochastic processes (MMs, HMMs) are explained explored, as are contemporary ML approaches including basic RNNs/LSTMs, CNNs, GANs, SEQ2SEQ, PIX2PIX, Word2Vec and other emerging methods.

* Understanding and applying advanced ML methods (Classification, Regression, Clustering, Sequence Matching)
* Building ML systems for problem solving (Gesture, Image, Sound and Text classification / tracking)
* ML systems for Creative Applications – image generation, sound generation

### Language: 
* Python 3.0+

#### Essential Reading 

* Karparthy, A, Hacker’s guide to Neural Networks - http://karpathy.github.io/neuralnets/

* Mital, P Creative Applications of Deep Learning in Tensor Flow - https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info

#### Further Reading 

* Karparthy, A, The Unreasonable Effectiveness of Recurrent Neural Networks - http://karpathy.github.io/2015/05/21/rnn-effectiveness/

* Whittaker, M. Crawford, K. et al. (2018).  AI Now Report. https://ainowinstitute.org/AI_Now_2018_Report.pdf

#### Grading: 

* Online testing: Students will complete a set of problem-solving assignments via online testing and video documentation of running code, which will be submitted for grading at the end of each session. (50%)  

* Final examination: Students will sit a final online written exam which will be 2 hours in duration. The exam comprises a 20 question multiple-choice test, and a series of three questions to test their knowledge of ML. Student’s will be asked to propose and describe appropriate solutions for given machine learning problems. (50%)

#### Web References:

https://www.kadenze.com/courses/machine-learning-for-musicians-and-artists/info

http://www.wekinator.org

http://www.rapidmixapi.com

http://ainowinstitute.org

https://ml4a.github.io/ml4a

### Weekly Outline: 

---

#### Week 1 (17.4.) - Intro + Motivation - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week01_intro-motivation/ml01_intro-motivation.pdf)

* **Recordings:** [lecture video](https://youtu.be/b6bjeelzB5c) (1:10h) and recording from our [practical session](https://youtu.be/9fox3RjL8Go) (47m)

* **Reading link:** _"CAN: Creative Adversarial Networks ..."_ (2017) - [pdf on arXiv](https://arxiv.org/abs/1706.07068)

<p align="center">
<a href="https://youtu.be/b6bjeelzB5c"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week01_intro-motivation/ml01_slide.gif" width="760"></a>
</p>

* **Topics:** Lecture showcases some cool examples of Machine Learning (in and outside of art context), concept of having a model and treating it as a black box in a processing pipeline. Practical section contains examples with coding up Linear Regression for a simple dataset of points.

---

#### Week 2 (24.4.) - Building blocks of NNs // Practicum: first models with Keras - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week02_basic-building-blocks/ml02_basic-building-blocks.pdf)

* **Recordings:** [lecture video](https://youtu.be/ptj4uIwsQtE) (1:23h) and recording from our [practical session](https://youtu.be/lqJ5L8nfFsI) (1:27h)

<p align="center">
<a href="https://youtu.be/ptj4uIwsQtE"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week02_basic-building-blocks/ml02_slide.gif" width="760"></a>
</p>

* **Topics:** Lecture explores the basic building blocks used when creating Neural Networks. Artificial Neurons, Neural Networks, connecting image data with Fully Connected NNs and finally training. In the practical session we create the neural network we studied about in the class (and show it's really simple to do so!) and train it on the MNIST dataset. We also show how to use (and as a bonus also how the hack) the trained model.

---

#### Week 3 - Convolutions, AlexNet, ImageNet // Practicum: using existing models - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week03_convolutional-nns/ml03_convolutional-nns.pdf)

* **Recordings:** [lecture video](https://youtu.be/cVU6WDyBXc4) (1:39h) and recording from our [practical session](https://youtu.be/-Z2OZMePgHo) (1:02h)

* **Reading link:** _"ImageNet Classification with Deep Convolutional Neural Networks"_ (2012) - [pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

<p align="center">
<a href="https://youtu.be/cVU6WDyBXc4"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week03_convolutional-nns/ml03_slide.gif" width="760"></a>
</p>

* **Topics:** Lecture starts with illustration of overfitting, then explores Convolutional operation applied over images. We then use convolutions inside Neural Networks and introduce Convolutional Neural Networks and their general model architecture. We describe the functionally differentiated sections of the model such as feature extractor and classifier. We visualize convolutional kernels to bring in some intuitive understanding to networks using convolutions (for tasks of classification or image generation). In the practical session we use existing trained neural networks to label images. We also extract feature descriptors of images inside a dataset (CIFAR-10) navigate the latent space (for searching and tracking trajectories).

---

#### Week 4 - Bank holiday

---

#### Week 5 - Interaction with Machine Learning models with Rebecca Fiebrink

Topics will include:

* Understanding how classification and regression can be used to create real-time interactions with sensors, audio, video, etc.
* Familiarity with "interactive machine learning" in which training data is iteratively modified to steer model behaviours
* Using Wekinator and MIMIC tools to train real-time interactive machine learning systems
* Using OSC to support real-time communication between applications/computers

##### Before end of day on 13 May, please do the following:

1. Install Wekinator for your operating system: http://www.wekinator.org/downloads/

2. Next, follow the instructions for the Wekinator “Quick Walkthrough” at http://www.wekinator.org/walkthrough/
Specifically:

* If you already have the Processing IDE installed on your computer, I recommend running the Processing source code examples. This is because you will easily be able to edit the examples to make them do other things. You can watch this video to learn how to add the necessary libraries in Processing: https://www.youtube.com/watch?embed=no&v=bE2EimjdUmM
* If you don’t want to run from the Processing source code, and you’re on a recent version of Windows, you may find that the webcam face tracker input, while more fun, doesn’t work— so try using the mouse-dragged box input instead.
* If you run into problems running one of the input or output executables, you may need to download the “last resort” files which are much larger but have the Java virtual machine included
* Read the text instructions closely to do the walkthrough.
* If you are able to get your chosen input (i.e., mouse-dragged box or webcam) and output (i.e., sound synthesis or colour change) to run, and you’ve followed all the instructions carefully, but things still don’t seem to be working right the problem is likely to be the fact that, at one point, you had more than one program trying to receive OSC at the same port at the same time. Quit all of your inputs, outputs, and Wekinator. Then open only your chosen input, your chosen output, and Wekinator, and follow the instructions again. This is likely to work.

If you run into any problems running Wekinator or doing the walkthrough, please send Rebecca a slack message or email her at r.fiebrink@arts.ac.uk and she’ll help you troubleshoot.

##### Lecture Videos

Please visit Moodle at https://moodle.arts.ac.uk/course/view.php?id=38156#section-5 for instructions for viewing the lecture videos. There are about 2 hours of video content to watch before Friday's practical session.

---

#### Week 6 - Beyond classification: Generative models, Domain to domain transfer - VAE, Pix2pix, Style transfer // Practicum: Scraping the internet

#### Week 7 - GANs // Practicum: Working with trained models in notebooks (google colab)

#### Week 8 - Sequential modelling and representing data

#### _PS: the order of the class topics might still be changed_
