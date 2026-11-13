/**********************************************************************
  TPC kernel: add 1 to every element of input.
  Based on canonical relu6_fwd.h structure (FP32 variant).
  Uses v_mov_s to broadcast the scalar 1.0 to a vector, then v_add_v_v.
 **********************************************************************/

#define FLOAT32
#include "kernel_config.h"

void main(tensor input, tensor output)
{
  const int depth = 0;
  const int width = 1;
  const int height = 2;
  const int batch = 3;
  const int fifthDim = 4;

  const int5 indexSpaceStart = get_index_space_offset();
  const int5 indexSpaceEnd = get_index_space_size() + indexSpaceStart;

  int5 ifmCoords = {0, 0, 0, 0, 0};

  // DEPTH
  const int depthStep = VECTOR_SIZE;
  const int depthStart = indexSpaceStart[depth] * depthStep;
  const int depthEnd = indexSpaceEnd[depth] * depthStep;

  // WIDTH
  const int widthStep = 4;
  const int widthStart = indexSpaceStart[width] * widthStep;
  const int widthEnd = indexSpaceEnd[width] * widthStep;

  // HEIGHT
  const int heightStep = 1;
  const int heightStart = indexSpaceStart[height];
  const int heightEnd = indexSpaceEnd[height];

  // BATCH
  const int batchStep = 1;
  const int batchStart = indexSpaceStart[batch];
  const int batchEnd = indexSpaceEnd[batch];

  // fifthDim
  const int fifthDimStep = 1;
  const int fifthDimStart = indexSpaceStart[fifthDim];
  const int fifthDimEnd = indexSpaceEnd[fifthDim];

  // Broadcast scalar 1.0 to a vector for adding 1 to every element
  VECTOR x00;
  const VECTOR one = v_mov_s((SCALAR)1.0);

#pragma loop_taken
  for (int d = depthStart; d < depthEnd; d += depthStep) {
    ifmCoords[depth] = d;

#pragma loop_taken
    for (int f = fifthDimStart; f < fifthDimEnd; f += fifthDimStep) {
      ifmCoords[fifthDim] = f;

#pragma loop_taken
      for (int b = batchStart; b < batchEnd; b += batchStep) {
        ifmCoords[batch] = b;

#pragma loop_taken
        for (int h = heightStart; h < heightEnd; h += heightStep) {
          ifmCoords[height] = h;

#pragma loop_taken
#pragma unroll 4
          for (int w = widthStart; w < widthEnd; w += 1) {
            ifmCoords[width] = w;

            x00 = v_ld_tnsr_i(ifmCoords, input);
            // Add 1 to every element
            x00 = v_add_v_v(x00, one, 0);
            // Store result
            st_tnsr_i_v(ifmCoords, output, x00);
          }
        }
      }
    }
  }
}
