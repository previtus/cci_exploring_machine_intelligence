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

#### Week 3 (1.5.) - Convolutions, AlexNet, ImageNet // Practicum: using existing models - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week03_convolutional-nns/ml03_convolutional-nns.pdf)

* **Recordings:** [lecture video](https://youtu.be/cVU6WDyBXc4) (1:39h) and recording from our [practical session](https://youtu.be/-Z2OZMePgHo) (1:02h)

* **Reading link:** _"ImageNet Classification with Deep Convolutional Neural Networks"_ (2012) - [pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

<p align="center">
<a href="https://youtu.be/cVU6WDyBXc4"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week03_convolutional-nns/ml03_slide.gif" width="760"></a>
</p>

* **Topics:** Lecture starts with illustration of overfitting, then explores Convolutional operation applied over images. We then use convolutions inside Neural Networks and introduce Convolutional Neural Networks and their general model architecture. We describe the functionally differentiated sections of the model such as feature extractor and classifier. We visualize convolutional kernels to bring in some intuitive understanding to networks using convolutions (for tasks of classification or image generation). In the practical session we use existing trained neural networks to label images. We also extract feature descriptors of images inside a dataset (CIFAR-10) navigate the latent space (for searching and tracking trajectories).

---

#### Bank holiday

---

#### Week 4 (15.5.) - Interaction with Machine Learning models with Rebecca Fiebrink

Topics will include:

* Understanding how classification and regression can be used to create real-time interactions with sensors, audio, video, etc.
* Familiarity with "interactive machine learning" in which training data is iteratively modified to steer model behaviours
* Using Wekinator and MIMIC tools to train real-time interactive machine learning systems
* Using OSC to support real-time communication between applications/computers

##### Before end of day on 13 May, please do the following:

1. Install Wekinator for your operating system: http://www.wekinator.org/downloads/

2. Next, follow the instructions for the Wekinator “Quick Walkthrough” at http://www.wekinator.org/walkthrough/
Specifically:

* For the input: If you're running OS X Catalina, there is currently a compatibility issue with Processing video and Catalina that will prevent the Processing face tracker from running. I recommend instead running a version of the face tracker (uses 3 inputs) in openFrameworks as your input, and choosing one of the 2 outputs from the walkthrough (they don't use the camera, so they should be fine). For the oF face tracker, you can get the source code in openFrameworks (http://www.doc.gold.ac.uk/~mas01rf/WekinatorDownloads/wekinator_examples/all_source_zips/FaceTracker_3Inputs_oF.zip) or just run the Mac executable (http://www.doc.gold.ac.uk/~mas01rf/WekinatorDownloads/wekinator_examples/executables/mac/inputs/FaceTracker_3Inputs_oF.zip) (Note: If the executable doesn’t put sunglasses on your face when you run it, drag the app to a new location (e.g., your desktop) then back. Make sure the app and the data/ directory share the same parent directory. Run it again. This is due to a peculiarity in security settings on OS X 10.12 and higher.) 
* If you already have the Processing IDE installed on your computer, I recommend running the Processing source code examples rather than the executables (with the exception of Catalina users trying to run the video input, as discussed above). This is because you will easily be able to edit the examples to make them do other things. You can watch this video to learn how to add the necessary libraries in Processing: https://www.youtube.com/watch?embed=no&v=bE2EimjdUmM
* If you don’t want to run from the Processing source code, and you’re on a recent version of Windows, you may find that the webcam face tracker input, while more fun, doesn’t work— so try using the mouse-dragged box input instead.
* If you run into problems running one of the input or output executables, you may need to download the “last resort” files which are much larger but have the Java virtual machine included
* Read the text instructions closely to do the walkthrough.
* If you are able to get your chosen input (i.e., mouse-dragged box or webcam) and output (i.e., sound synthesis or colour change) to run, and you’ve followed all the instructions carefully, but things still don’t seem to be working right the problem is likely to be the fact that, at one point, you had more than one program trying to receive OSC at the same port at the same time. Quit all of your inputs, outputs, and Wekinator. Then open only your chosen input, your chosen output, and Wekinator, and follow the instructions again. This is likely to work.

If you run into any problems running Wekinator or doing the walkthrough, please send Rebecca a slack message or email her at r.fiebrink@arts.ac.uk and she’ll help you troubleshoot.

##### Lecture Videos

Please visit Moodle at https://moodle.arts.ac.uk/course/view.php?id=38156#section-5 for instructions for viewing the lecture videos. There are about 2 hours of video content to watch before Friday's practical session.

---

#### Week 5 (22.5.) - Beyond classification: Generative models - Variational AutoEncoders, Generative Adversarial Networks // Practicum: Scraping the internet - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week05_generative-models/ml05_generative-models.pdf)

* **Recordings:** [lecture video](https://youtu.be/QFy8Z0tRhy0) (1:31h) and recording from our [practical session](https://youtu.be/saDUunZ7eqQ) (1:12h)

<p align="center">
<a href="https://youtu.be/QFy8Z0tRhy0"><img src="https://raw.githubusercontent.com/previtus/cci_exploring_machine_intelligence/master/week05_generative-models/ml05_slide.gif" width="760"></a>
</p>

* **Topics:** This lecture is about Generative models. We start with explaining the AutoEncoder (AE) architecture and explore the ways in which we can interact with these models. One of the AE specific interactions is the possibility to encode real images into latent space and in doing so extract the visual attribute vectors (can you imagine a vector responsible for "smile" in a human faces dataset?). In a second part of the lecture we dig deeper into using Generative Adversarial Networks (GANs), explain the differences with AEs and reasons to use one or the other type of architecture. We look into some _weird_, cool and non-traditional uses of these models in art projects. In the practical session we learn how to create, train and use a VAE model with some basic datasets. We will also talk about collecting your own datasets by scraping the internet.

---

#### Week 6 - Additional generative models: pix2pix, style transfer and deep dreaming // Practicum: Practical session with Progressively Growing GAN on Google Colab - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week06_generative-models-ii/ml06_generative-models-II.pdf)

* **Recordings:** [lecture video](https://youtu.be/qmOESGWRPe4) (52m)

<p align="center">
<a href="https://youtu.be/qmOESGWRPe4"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week06_generative-models-ii/ml06_slide.gif" width="760"></a>
</p>

* **Topics:** This is our second lecture focused on Generative models. We talk through some additional machine learning techniques, namely the: domain to domain translation (pix2pix model), style transfer and finally the deep dream algorithm. This week we focus more on the practical session, where we go through all the necessary steps needed to train and use a ProgressiveGAN model on your own dataset (this is shown on Colab).

* **Seminar session focused on Progressive GAN:** [practical session](https://youtu.be/hEwuhfWZqkI) (1:46h)

<p align="center">
<a href="https://youtu.be/hEwuhfWZqkI"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week06_generative-models-ii/ml06practicum_slide.gif" width="760"></a>
</p>

---

#### Week 7 - Sequential data modelling // Practicum: Q&A for Final Exam - [slides pdf](https://github.com/previtus/cci_exploring_machine_intelligence/blob/master/week07_sequential-modelling/ml07_sequential-modelling.pdf)

* **Recordings:** [lecture video](https://youtu.be/QVEkOFqK9MY) (45m)

<p align="center">
<a href="https://youtu.be/QVEkOFqK9MY"><img src="https://github.com/previtus/cci_exploring_machine_intelligence/raw/master/week07_sequential-modelling/ml07_slide.gif" width="760"></a>
</p>

* **Topics:** This lecture is about modelling sequential data with some specialized models which include sequentiality in their design. We present the Recurrent Neural Network (RNN) model and the Long short-term memory (LSTM) model. We show how to plug in data from two domains - textual data encoded as one-hot vectors using a dictionary and musical data encoded using the Fourier Transform into spectrogram representation. Our live practical session will be opened for a Q&A about the final exams.
