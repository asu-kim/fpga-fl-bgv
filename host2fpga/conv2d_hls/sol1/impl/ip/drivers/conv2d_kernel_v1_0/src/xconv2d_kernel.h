// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef XCONV2D_KERNEL_H
#define XCONV2D_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xconv2d_kernel_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
#ifdef SDT
    char *Name;
#else
    u16 DeviceId;
#endif
    u64 Control_BaseAddress;
} XConv2d_kernel_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XConv2d_kernel;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XConv2d_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XConv2d_kernel_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XConv2d_kernel_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XConv2d_kernel_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
#ifdef SDT
int XConv2d_kernel_Initialize(XConv2d_kernel *InstancePtr, UINTPTR BaseAddress);
XConv2d_kernel_Config* XConv2d_kernel_LookupConfig(UINTPTR BaseAddress);
#else
int XConv2d_kernel_Initialize(XConv2d_kernel *InstancePtr, u16 DeviceId);
XConv2d_kernel_Config* XConv2d_kernel_LookupConfig(u16 DeviceId);
#endif
int XConv2d_kernel_CfgInitialize(XConv2d_kernel *InstancePtr, XConv2d_kernel_Config *ConfigPtr);
#else
int XConv2d_kernel_Initialize(XConv2d_kernel *InstancePtr, const char* InstanceName);
int XConv2d_kernel_Release(XConv2d_kernel *InstancePtr);
#endif

void XConv2d_kernel_Start(XConv2d_kernel *InstancePtr);
u32 XConv2d_kernel_IsDone(XConv2d_kernel *InstancePtr);
u32 XConv2d_kernel_IsIdle(XConv2d_kernel *InstancePtr);
u32 XConv2d_kernel_IsReady(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_Continue(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_EnableAutoRestart(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_DisableAutoRestart(XConv2d_kernel *InstancePtr);

void XConv2d_kernel_Set_enc_weights(XConv2d_kernel *InstancePtr, u64 Data);
u64 XConv2d_kernel_Get_enc_weights(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_Set_enc_bias(XConv2d_kernel *InstancePtr, u64 Data);
u64 XConv2d_kernel_Get_enc_bias(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_Set_enc_input(XConv2d_kernel *InstancePtr, u64 Data);
u64 XConv2d_kernel_Get_enc_input(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_Set_enc_output(XConv2d_kernel *InstancePtr, u64 Data);
u64 XConv2d_kernel_Get_enc_output(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_Set_rows(XConv2d_kernel *InstancePtr, u32 Data);
u32 XConv2d_kernel_Get_rows(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_Set_cols(XConv2d_kernel *InstancePtr, u32 Data);
u32 XConv2d_kernel_Get_cols(XConv2d_kernel *InstancePtr);

void XConv2d_kernel_InterruptGlobalEnable(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_InterruptGlobalDisable(XConv2d_kernel *InstancePtr);
void XConv2d_kernel_InterruptEnable(XConv2d_kernel *InstancePtr, u32 Mask);
void XConv2d_kernel_InterruptDisable(XConv2d_kernel *InstancePtr, u32 Mask);
void XConv2d_kernel_InterruptClear(XConv2d_kernel *InstancePtr, u32 Mask);
u32 XConv2d_kernel_InterruptGetEnabled(XConv2d_kernel *InstancePtr);
u32 XConv2d_kernel_InterruptGetStatus(XConv2d_kernel *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
