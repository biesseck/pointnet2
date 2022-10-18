#/bin/bash

# CUDA_HOME=/usr/local/cuda-10.0
CUDA_HOME=/usr/local/cuda-8.0

# Bernardo
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')


# TF1.2
# g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0



# TF1.4
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Bernardo
g++ -std=c++14 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I $TF_INC -I $TF_INC/external/nsync/public -I $CUDA_HOME/include -lcudart -ltensorflow_framework -L $CUDA_HOME/lib64/ -L$TF_LIB -O2 -D_GLIBCXX_USE_CXX11_ABI=0
