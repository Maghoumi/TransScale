TransScale
==========

Provides transparent scalability for using multiple GPUs with JCuda.
TransScale facilitates agile CUDA development in Java using JCuda. With TransScale, it is now
easy to load modules on single or multiple CUDA-enabled GPUs and harness their processing power.

TransScale also provides wrappers for 2D CUDA primitive arrays of structs with arbitrary number
of fields (such as a 2D array of char's representing an array of colors in ARGB space). These
2D allocations can be automatically pitched if possible and the pitch values are converted to a
number that can be treated as a regular "width" similar to any 2D allocated array. The wrapped
pointers are extensions of CUdeviceptr which allows transparent manipulation using JCuda.

Note that the project is still very immature and more features and samples may be added in the future. 