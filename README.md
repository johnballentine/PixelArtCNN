<h1>PixelArtCNN: Convolutional neural network for sprite restoration.</h1>

This project aims to restore distorted pixel art images to a pixel-perfect state. Pixel art, a distinctive form of digital art known for its clean lines, distinct color blocks, and unique palettes, frequently encounters issues such as distortion caused by JPEG compression, multiple iterations of upscaling, and other transformations that occur during the distribution of these images on the internet. The transparency mask (or alpha channel) is also usually lost when rendering the images for distribution. Additionally, text-to-image diffusion models such as Midjourney and Stable Diffusion are trained on these images, therefore their outputs contain the same imperfections.

Conventional image processing methods often struggle to precisely restore the distinct characteristics of pixel art. The diverse range of distortion causes in source images, combined with the relatively constrained robustness of these techniques, imposes limits on their performance for this task. By leveraging a convolutional neural network and a custom loss function that penalizes edge blurriness, PixelArtCNN can restore the transparency mask, the distinct color palettes, and the sharply defined boundaries that define the pixel art style.

<h3>Model Architecture</h3>

The model adopts an encoder-only architecture that applies convolutional operations, ReLU activation, and max-pooling to progressively reduce the resolution of the input image from 256x256 to 16x16 while increasing the number of feature channels to preserve essential features during the downscaling process.
<br>
<br>
<center><img src="images/diagram.png" alt="Architecture Diagram Image" width="75%"></center>
<br>
<br>
Each convolutional block reduces the resolution of the input by a factor of 2 in both spatial dimensions. This downsampling is achieved through the use of max pooling, which selects the maximum value within a sliding window and discards the rest.

The choice of using an encoder-only architecture is based on the specific requirements of pixel art. Preserving the pixel-perfect nature of the images is crucial. By avoiding any upscaling steps in the model architecture, we prevent the loss of visual integrity.

In the final block, a 1x1 convolution layer is used to transition from 512 feature channels down to 4 channels, mapping the feature representations into the desired RGBA channel configuration while preserving spatial resolution. A sigmoid activation function is then applied to normalize the output values to the [0,1] range.

<h3>Usage</h3>

usage goes here