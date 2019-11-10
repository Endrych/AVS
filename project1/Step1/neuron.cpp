/*
 * Architektury výpočetních systémů (AVS 2019)
 * Projekt c. 1 (ANN)
 * Login: xlogin00
 */

#include <cstdlib>
#include "neuron.h"
#include <iostream>

float evalNeuron(
  size_t inputSize,
  size_t neuronCount,
  const float* input,
  const float* weights,
  float bias,
  size_t neuronId
)
{
  //TODO: Step0 - Fill in the implementation, all the required arguments are passed.
  //              If you don't use them all you are doing something wrong!
    float sum = 0;
    for(int i=0;i<inputSize;i++){
      sum += input[i] * weights[i * neuronCount + neuronId];
    }

    sum += bias;
    if(sum < 0){
      return 0;
    }

  return sum;
}
