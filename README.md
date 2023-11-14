# Parallel Patch Match

## Resources
https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1211538

https://cave.cs.columbia.edu/old/publications/pdfs/Kumar_ECCV08_2.pdf

https://people.engr.tamu.edu/nimak/Data/ICCP14_MaskedPatches.pdf

https://cs.brown.edu/courses/csci1290/2011/asgn/proj3/

## Title: Parallel NNFs with PatchMatch
Micah Reich (mreich), David Krajewski (dkrajews)
## Summary
We are going to implement a parallelized version of the PatchMatch algorithm for nearest-neighbor field (NNF) generation on GPU and CPU. NNFs can then be used to perform image inpainting or content-aware fill as well as optical flow for target tracking in video. Many of the other image operations within inpainting can also be handled in a data parallel fashion.

## Background
Before the advent of modern neural network architectures and commonplace GPUs on consumer hardware, applications like Adobe PhotoShop had content-aware fill options for image inpainting which ran on CPUs. Many of the methods for inpainting relied on NNFs to match image regions to similar regions outside the fill region. These NNFs were constructed in different ways, ranging from tree-based acceleration structures like kD-Trees, PCA trees, Ball trees, or VP trees to accelerate the search for nearest neighbor patches [1](https://cave.cs.columbia.edu/old/publications/pdfs/Kumar_ECCV08_2.pdf). Later, a randomized algorithm known as PatchMatch [2](https://gfx.cs.princeton.edu/pubs/Barnes_2009_PAR/patchmatch.pdf) was developed to efficiently find NNFs without the memory overhead of acceleration structures in near realtime. 

PatchMatch works by 

## The Challenge

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
Content-aware fill with drawn-on masks

### Stretch Goals
Frame-to-frame PatchMatch for object tracking within drawn-on masks

## Platform Choice
We have chosen an NVIDIA RTX 2080 GPU running CUDA and an Intel i7-9700 8-core CPU running OpenMP. It makes sense to use these parallel systems because the PatchMatch algorithm has independent work in each NNF iteration as well as other image operations (blending, smoothing) which can be done in a data-parallel fashion. GPUs are notoriously successful in parallelizing image operations, and we can similarly parallelize NNF construction across CPU cores, albeit to a lesser degree.
