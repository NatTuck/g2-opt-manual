/**********************************************************************
  Glue code header for add1_fwd_f32_gaudi2 TPC kernel.
 **********************************************************************/

#ifndef _ADD1_FWD_F32_GAUDI2_HPP
#define _ADD1_FWD_F32_GAUDI2_HPP

#include "gc_interface.h"
#include "tpc_kernel_lib_interface.h"

class Add1FwdF32Gaudi2
{
public:
    Add1FwdF32Gaudi2() {}
    virtual ~Add1FwdF32Gaudi2() {}

    virtual tpc_lib_api::GlueCodeReturn
    GetGcDefinitions(tpc_lib_api::HabanaKernelParams* in_defs,
                    tpc_lib_api::HabanaKernelInstantiation* out_defs);

    virtual tpc_lib_api::GlueCodeReturn GetKernelName(
            char kernelName [tpc_lib_api::MAX_NODE_NAME]);
};

#endif
