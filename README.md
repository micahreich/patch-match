# Parallel Patch Match

## Resources
https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1211538 (pg 2+)

https://cave.cs.columbia.edu/old/publications/pdfs/Kumar_ECCV08_2.pdf

https://people.engr.tamu.edu/nimak/Data/ICCP14_MaskedPatches.pdf

https://cs.brown.edu/courses/csci1290/2011/asgn/proj3/

https://www.ipol.im/pub/art/2017/189/article.pdf

## Title: Parallel NNFs with PatchMatch
Micah Reich (mreich), David Krajewski (dkrajews)
## Summary
We are going to implement a parallelized version of the PatchMatch algorithm for nearest-neighbor field (NNF) generation on GPU and CPU. NNFs can then be used to perform image inpainting or content-aware fill as well as optical flow for target tracking in video. Many of the other image operations within inpainting can also be handled in a data parallel fashion.

## Background
Before the advent of modern neural network architectures and commonplace GPUs on consumer hardware, applications like Adobe PhotoShop had content-aware fill options for image inpainting which ran on CPUs. Many of the methods for inpainting relied on NNFs to match image regions to similar regions outside the fill region. These NNFs were constructed in different ways, ranging from tree-based acceleration structures like kD-Trees, PCA trees, Ball trees, or VP trees to accelerate the search for nearest neighbor patches [1](https://cave.cs.columbia.edu/old/publications/pdfs/Kumar_ECCV08_2.pdf). Later, a randomized algorithm known as PatchMatch [2](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf) was developed to efficiently find NNFs without the memory overhead of acceleration structures in near realtime. 

The PatchMatch algorithm at a high level attempts to find corresponding "patches" across two images. The paper defines NNFs as a function that maps the coordinates of a patch in one image to the offset of this patch in the second image. We begin by randomly initializing these offsets uniformly. We then get to the iteration phase. The algorithm, for reasons that will be obvious, iterates "in order". That is, it goes from left to right, up to down, when computing the offsets. To compute these offsets, we alternate between a propagation step and a random search step. The propagation step leverages the fact that "nearby" offsets (i.e. the offset corresponding to the patch left and above it) have been computed successfully. Since these nearby offsets should in theory be close to the offset we are currently trying to compute, we can use them as a good guess for where our patch should be. The other step is the random search step. This step involves randomly searching nearby to where we think the current offset is to hopefully find a better match. This algorithm repeats for several iterations (the paper suggests 4-5). 

In terms of what we can parallelize, the most obvious step is the random search step. Because that one is totally independent of the other patches, we can split the random search step up and completely parallelize it across all patches at once. An interesting challenge will be parallelizing the propagation step. Because there are some dependencies, we might consider doing some sort of red-black ordering and parallelizing that way, but we are unsure and will keep exploring our options.

## The Challenge
A major challenge in implementing a parallelized version of PatchMatch is having to work around the propagation step contains data dependencies that bottleneck the performance. Identifying and implementing techniques to try and limit some of these data dependencies across iterations while maintaining high performance and fast convergence will be difficult. 
By the nature of how the algorithm works, we anticipate there being heavy locality across iterations. For example, computing the offset of a patch during the propagation step relies on nearby patches, which are physically close on the image and therefore could exhibit good spatial locality. 

## Resources
We will start from scratch, first developing a sequential algorithm which performs image inpainting / content-aware fill in images. We are using a few papers as reference for the procedure of hole-filling as well as the specifics of the PatchMatch algorithm:

1. https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf
1. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1211538
1. https://cave.cs.columbia.edu/old/publications/pdfs/Kumar_ECCV08_2.pdf
1. https://people.engr.tamu.edu/nimak/Data/ICCP14_MaskedPatches.pdf
1. https://cs.brown.edu/courses/csci1290/2011/asgn/proj3/

For compute resources, we will need access to an NVIDIA GPU as well as a multi-core CPU. Both of these can be accessed through the GHC clusters used for previous assignments in the course. Specifically, we will use an Intel i7-9700 8-core CPU and an NVIDIA RTX 2080 to benchmark results.

## Goals and Deliverables
### Plan to Achieve
Content-aware fill with drawn-on masks.
For the poster session, we hope to have an interface linked to a backend running our algorithm.

### Stretch Goals
Frame-to-frame PatchMatch for object tracking within drawn-on masks

## Platform Choice
We have chosen an NVIDIA RTX 2080 GPU running CUDA and an Intel i7-9700 8-core CPU running OpenMP. It makes sense to use these parallel systems because the PatchMatch algorithm has independent work in each NNF iteration as well as other image operations (blending, smoothing) which can be done in a data-parallel fashion. GPUs are notoriously successful in parallelizing image operations, and we can similarly parallelize NNF construction across CPU cores, albeit to a lesser degree.

## Schedule
Week of November 13th:
- Have a sequential version of the algorithm working in Python

Week of November 20th:
- Finish a parallel CPU implementation with C++ and OpenMP

Week of November 27th: 
- Finish CUDA implementation of the algorithm

Week of December 4th:
- Create a basic web interface that allows for image uploading
- Setup an API for our algorithm

Week of December 11th:
- Have a working demo with an interface
