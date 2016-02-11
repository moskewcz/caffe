#include <vector>

#include "caffe/layers/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


  static uint32_t det_drop_seed = 0;

  extern "C" void caffe_set_det_drop_seed( uint32_t const det_drop_seed_ ) { det_drop_seed = det_drop_seed_; }

// fmix32 here is from: https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp (public domain)
inline __device__ void fmix32( uint32_t & h ) {
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
}


template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
			       Dtype* out, uint32_t const det_drop_seed ) {
  CUDA_KERNEL_LOOP(index, n) {
    bool set_v = 0;
    if( det_drop_seed == 0 ) { if( mask[index] > threshold ) { set_v = 1; } }
    else {
      unsigned h = index + det_drop_seed;
      fmix32(h);
      if( h > threshold ) { set_v = 1; }
    }
    out[index] = set_v ? (in[index] * scale) : Dtype(0);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    if( !det_drop_seed ) { caffe_gpu_rng_uniform(count, mask); } // FIXME: mask unused if det_drop_seed != 0
    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, mask, uint_thres_, scale_, top_data, det_drop_seed );
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
				Dtype* out_diff, uint32_t const det_drop_seed ) {
  CUDA_KERNEL_LOOP(index, n) {
    bool set_v = 0;
    if( det_drop_seed == 0 ) { if( mask[index] > threshold ) { set_v = 1; } }
    else {
      unsigned h = index + det_drop_seed;
      fmix32(h);
      if( h > threshold ) { set_v = 1; }
    }
    out_diff[index] = set_v ? (in_diff[index] * scale) : Dtype(0);
    //out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      // NOLINT_NEXT_LINE(whitespace/operators)
      DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, mask, uint_thres_, scale_, bottom_diff, det_drop_seed );
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
