# English to Chinese Translation using Transformer architecture

Simply a training script and brief test program for sequence-to-sequence translation between English and Mandarin Chinese.

If you want to use this you will need to make a new directory `data/` and copy the dataset to this directory, naming it `cmn.txt`.
Training data is taken from [here](https://www.manythings.org/anki/)

As it stands, this script will only work for translation from languages with clearly delimited words to less clearly delimited
languages such as Chinese, Japanese, etc. This is due to me hard coding in some values which can easily be changed later.
The separation of words in such languages is still a bit of a open problem which I do not attempt to tackle here.

Additionally, a couple of Traditional Chinese characters have made their way into the dataset. This may make some results different to what is expected but
should (hopefully) be equivalent meaning.

This model, by default, takes just under one minute per epoch on a GTX 1080Ti with a batch size of 64.
Increasing the embedding dimensions will (somewhat obviously) increase this time. I haven't found increasing the size to be hugely beneficial though.

This repository was mainly born out of frustration at the PyTorch tutorials for Transformer architecture- in particular [Sequence-to-sequence modeling with nn.Transformer and torchtext](https://pytorch.org/tutorials/beginner/transformer_tutorial.html). This is because, rather amusingly, it does not actually use the nn.Transformer module despite the title. It only uses half of the full Transformer architecture (the encoder) defined in [Attention is All You Need (Vaswani et. al.)](https://arxiv.org/abs/1706.03762) even though (again, amusingly) it has a diagram of the full architecture in the tutorial. Hence out of this frustration I made this repository to teach myself how to use it.

There is an open issue on GitHub for aforementioned frustration which can be found [here](https://github.com/pytorch/tutorials/issues/719).
