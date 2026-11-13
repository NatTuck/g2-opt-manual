/**********************************************************************
  Entry points for libadd1_kernels.so.
  Exports GetKernelGuids and InstantiateTpcKernel.
 **********************************************************************/

#include "add1_fwd_f32_gaudi2.hpp"
#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"
#include <stdio.h>
#include <string.h>

#define GAUDI2_KERNEL_ADD1_FWD_F32 0
#define GAUDI2_KERNEL_MAX 1

extern "C" {

tpc_lib_api::GlueCodeReturn GetKernelGuids(
        _IN_ tpc_lib_api::DeviceId deviceId,
        _INOUT_ uint32_t* kernelCount,
        _OUT_ tpc_lib_api::GuidInfo* guids)
{
    if (deviceId != tpc_lib_api::DEVICE_ID_GAUDI2) {
        if (kernelCount != nullptr) *kernelCount = 0;
        return tpc_lib_api::GLUE_SUCCESS;
    }

    if (guids != nullptr) {
        Add1FwdF32Gaudi2 instance;
        instance.GetKernelName(guids[GAUDI2_KERNEL_ADD1_FWD_F32].name);
    }

    if (kernelCount != nullptr) {
        *kernelCount = GAUDI2_KERNEL_MAX;
    }
    return tpc_lib_api::GLUE_SUCCESS;
}

tpc_lib_api::GlueCodeReturn
InstantiateTpcKernel(
        _IN_ tpc_lib_api::HabanaKernelParams* params,
        _OUT_ tpc_lib_api::HabanaKernelInstantiation* instance)
{
    char kernelName[tpc_lib_api::MAX_NODE_NAME];

    Add1FwdF32Gaudi2 kernelInstance;
    kernelInstance.GetKernelName(kernelName);

    if (strcmp(params->guid.name, kernelName) == 0) {
        return kernelInstance.GetGcDefinitions(params, instance);
    }
    return tpc_lib_api::GLUE_NODE_NOT_FOUND;
}

tpc_lib_api::GlueCodeReturn GetShapeInference(
        tpc_lib_api::DeviceId deviceId,
        tpc_lib_api::ShapeInferenceParams* inputParams,
        tpc_lib_api::ShapeInferenceOutput* outputData)
{
    return tpc_lib_api::GLUE_SUCCESS;
}

}  // extern "C"
