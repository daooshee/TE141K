# Benchmarking Representative Style Transfer Models

Links to the papers and corresponding codes of all benchmarked models.

Parts borrowed from [[Neural Style Transfer: A Review]](https://github.com/ycjing/Neural-Style-Transfer-Papers)

## Global-based

1. **[NST]** L. A. Gatys, A. S. Ecker, and M. Bethge, “Image style transfer using convolutional neural networks,” in Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2016, pp. 2414–2423. [[Paper]](https://arxiv.org/pdf/1508.06576.pdf)

*   [Torch-based](https://github.com/jcjohnson/neural-style)
*   [TensorFlow-based](https://github.com/anishathalye/neural-style)
*   [TensorFlow-based with L-BFGS optimizer support](https://github.com/cysmith/neural-style-tf)
*   [Caffe-based](https://github.com/fzliu/style-transfer) 
*   [Keras-based](https://github.com/titu1994/Neural-Style-Transfer)
*   [MXNet-based](https://github.com/pavelgonchar/neural-art-mini)
*   [MatConvNet-based](https://github.com/aravindhm/neural-style-matconvnet)

2. **[AdaIN]** X. Huang and S. Belongie, “Arbitrary style transfer in real-time with adaptive instance normalization,” in Proc. Int’l Conf. Computer Vision, 2017, pp. 1510–1519. [[Paper]](https://arxiv.org/pdf/1703.06868.pdf)

*   [Torch-based](https://github.com/xunhuang1995/AdaIN-style)
*   [TensorFlow-based with Keras](https://github.com/eridgd/AdaIN-TF)
*   [TensorFlow-based without Keras](https://github.com/elleryqueenhomels/arbitrary_style_transfer)


3. **[WCT]** Y. Li, C. Fang, J. Yang, Z. Wang, X. Lu, and M.-H. Yang, “Universal style transfer via feature transforms,” in Advances in Neural Information Processing Systems, 2017, pp. 386–396. [[Paper]](https://arxiv.org/pdf/1705.08086.pdf)

*   [Torch-based](https://github.com/Yijunmaverick/UniversalStyleTransfer)
*   [TensorFlow-based](https://github.com/eridgd/WCT-TF)
*   [PyTorch-based #1](https://github.com/sunshineatnoon/PytorchWCT)
*   [PyTorch-based #2](https://github.com/pietrocarbo/deep-transfer)


## Local-based

1. **[Analogy]** A. Hertzmann, C. E. Jacobs, N. Oliver, B. Curless, and D. H. Salesin, “Image analogies,” in Proc. Conf. Computer Graphics and Interactive Techniques, 2001, pp. 327–340. [[Paper]](https://www.mrl.nyu.edu/publications/image-analogies/analogies-300dpi.pdf)

*   [C-based](https://mrl.nyu.edu/projects/image-analogies/)

2. **[Quilting]** A. A. Efros and W. T. Freeman, “Image quilting for texture synthesis and transfer,” in Proc. ACM Conf. Computer Graphics and Interactive Techniques, 2001, pp. 341–346. [[Paper]](http://graphics.cs.cmu.edu/people/efros/research/quilting/quilting.pdf)

*   [Matlab-based](https://github.com/PJunhyuk/ImageQuilting)

3. **[CNNMRF]** C. Li and M. Wand, “Combining markov random fields and convolutional neural networks for image synthesis,” in Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2016, pp. 2479–2486. [[Paper]](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Li_Combining_Markov_Random_CVPR_2016_paper.pdf)

*   [Torch-based](https://github.com/chuanli11/CNNMRF)

4. **[Doodles]**  A. J. Champandard, “Semantic style transfer and turning two-bit doodles into fine artworks,” 2016, arXiv:1603.01768. [[Paper]](https://arxiv.org/pdf/1603.01768.pdf) 

*   [Torch-based](https://github.com/alexjc/neural-doodle)

5. **[T-Effect]** S. Yang, J. Liu, Z. Lian, and Z. Guo, “Awesome typography: Statistics-based text effects transfer,” in Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2017, pp. 7464–7473. [[Paper]](https://arxiv.org/abs/1611.09026)

*   [Matlab-based](https://github.com/williamyang1991/Text-Effects-Transfer)

6. **[UT-Effect]** S. Yang, J. Liu, W. Yang, and Z. Guo, “Context-aware text-based binary image stylization and synthesis,” IEEE Transactions on Image Processing, vol. 28, no. 2, pp. 952–964, 2019. [[Paper]](https://arxiv.org/pdf/1810.03767.pdf)


## GAN-based

1. **[Pix2pix]** P. Isola, J. Y. Zhu, T. Zhou, and A. A. Efros, “Image-to-image translation with conditional adversarial networks,” in Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2017, pp. 5967–5976. [[Paper]](https://arxiv.org/pdf/1611.07004.pdf)

*   [Torch-based](https://github.com/phillipi/pix2pix)
*   [Pytorch-based](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
*   [TensorFlow-based](https://github.com/yenchenlin/pix2pix-tensorflow)
*   [Chainer-based](https://github.com/yenchenlin/pix2pix-tensorflow)

2. **[BicycleGAN]** J.-Y. Zhu, R. Zhang, D. Pathak, T. Darrell, A. A. Efros, O. Wang, and E. Shechtman, “Toward multimodal image-to-image translation,” in Advances in Neural Information Processing Systems, 2017, pp. 465–476. [[Paper]](https://arxiv.org/pdf/1711.11586.pdf)

*   [PyTorch-based](https://github.com/junyanz/BicycleGAN)
*   [TensorFlow-based #1](https://github.com/gitlimlab/BicycleGAN-Tensorflow)
*   [TensorFlow-based #2](https://github.com/kvmanohar22/img2imgGAN)

3. **[StarGAN]** Y. Choi, M. Choi, M. Kim, J. W. Ha, S. Kim, and J. Choo, “Stargan: Unified generative adversarial networks for multi-domain image-to-image translation,” in Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2018. [[Paper]](https://arxiv.org/pdf/1711.09020.pdf)

*   [PyTorch-based](https://github.com/yunjey/StarGAN)

4. **[MC-GAN]** S. Azadi, M. Fisher, V. Kim, Z. Wang, E. Shechtman, and T. Darrell, “Multi-content gan for few-shot font style transfer,” in Proc. IEEE Int’l Conf. Computer Vision and Pattern Recognition, 2018. [[Paper]](https://arxiv.org/pdf/1712.00516.pdf)

*   [PyTorch-based](https://github.com/azadis/MC-GAN)

5. **[TET-GAN]** (ours) [[AAAI Version Paper]](https://arxiv.org/pdf/1812.06384.pdf)

*   [PyTorch-based](https://github.com/williamyang1991/TET-GAN)
