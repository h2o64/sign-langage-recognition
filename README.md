# Sign Language Recognition - Notebooks



Here are the notebooks we designed during our project :

* `MNIST Data - Good model` describes our work on the [Sign MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
* `New data - Data preparation` describes our new homemade dataset and how we cleaned it
* `New data - First bad model` describes our homemade CNN for our new homemade dataset
* `New data - Finetuning of pretrained` describes our finetuning solution for our new homemade dataset

All the weights and intermediate saves are located in `saves/` except

* `finetuning_resnet.h5` which can be obtained [here](https://drive.google.com/file/d/1T3ETrBQ_yzMTROLXcgwSkRl44yli5Mwh/view?usp=sharing)
* `resized_images.npy` which can be obtained [here](https://drive.google.com/file/d/13I9oPf5lm58ql8kpWxmxi3YkMEJGjiP8/view?usp=sharing)

As for the data, the outputs of the `New data - Data preparation` can be obtained here 

* `new_data_filtered_splitted_v2.zip` [here](https://drive.google.com/file/d/1-8Dfy0s8I-vXZAGhXinCjRrw8rdsjixE/view?usp=sharing)
* `new_data_filtered_splitted_grouped_v3.zip` [here](https://drive.google.com/file/d/1JnRP7thN9_zwA3vbY3UDLn1KExHKaIJa/view?usp=sharing)

The full data won't be available.

A **proof of concept** with YOLO for live sign recognition is available on the [`yolo-detection`](https://github.com/h2o64/sign-language-recognition/tree/yolo-detection) branch (this is a fork from [`cansik/yolo-hand-detection`](https://github.com/cansik/yolo-hand-detection))