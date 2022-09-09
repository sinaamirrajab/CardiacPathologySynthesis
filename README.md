# Pathology synthesis of 3D-consistent cardiac MR images using 2D VAEs and GANs
Sina Amirrajab, Cristian Lorenz, Jurgen Weese, Josien Pluim, Marcel Breeuwer.
Eindhoven University of Technology, Biomedical Engineering Department, Medical Image Analysis group
<!-- ---- -->
## Abstract
We propose a method for synthesizing cardiac MR images with plausible heart shapes and realistic appearances for the purpose of generating labeled data for deep-learning (DL) training. It breaks down the image synthesis into label defor-mation and label-to-image translation tasks. The former is achieved via latent space interpolation in a VAE model, while the latter is accomplished via a condi-tional GAN model. We devise an approach for label manipulation in the latent space of the trained VAE model, namely pathology synthesis, aiming to synthe-size a series of pseudo-pathological synthetic subjects with characteristics of a desired heart disease. Furthermore, we propose to model the relationship between 2D slices in the latent space of the VAE via estimating the correlation coefficient matrix between the latent vectors and utilizing it to correlate elements of randomly drawn samples before decoding to image space. This simple yet effective ap-proach results in generating 3D consistent subjects from 2D slice-by-slice gen-erations. Such an approach could provide a solution to diversify and enrich the available database of cardiac MR images and to pave the way for the development of generalizable DL based image analysis algorithms.

<p align='center'>
  <img src='visuals/pathology_synthesis.png ' width='10000'/>
</p>
Pathology synthesis to generate the transition between a normal subject (NOR) to a target pathol-ogy such as dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM) and dilated right ventricle (RV). The effects of a disease on the heart geometry of a subject are respectively left ventricle dilation, myocardial thickening and right ventricle dilation.


## Code 
The code is being uploaded...

## Paper citation
Will be added...