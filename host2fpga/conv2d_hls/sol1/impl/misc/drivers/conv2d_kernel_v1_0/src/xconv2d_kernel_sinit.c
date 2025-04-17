// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#ifdef SDT
#include "xparameters.h"
#endif
#include "xconv2d_kernel.h"

extern XConv2d_kernel_Config XConv2d_kernel_ConfigTable[];

#ifdef SDT
XConv2d_kernel_Config *XConv2d_kernel_LookupConfig(UINTPTR BaseAddress) {
	XConv2d_kernel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = (u32)0x0; XConv2d_kernel_ConfigTable[Index].Name != NULL; Index++) {
		if (!BaseAddress || XConv2d_kernel_ConfigTable[Index].Control_BaseAddress == BaseAddress) {
			ConfigPtr = &XConv2d_kernel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XConv2d_kernel_Initialize(XConv2d_kernel *InstancePtr, UINTPTR BaseAddress) {
	XConv2d_kernel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XConv2d_kernel_LookupConfig(BaseAddress);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XConv2d_kernel_CfgInitialize(InstancePtr, ConfigPtr);
}
#else
XConv2d_kernel_Config *XConv2d_kernel_LookupConfig(u16 DeviceId) {
	XConv2d_kernel_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XCONV2D_KERNEL_NUM_INSTANCES; Index++) {
		if (XConv2d_kernel_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XConv2d_kernel_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XConv2d_kernel_Initialize(XConv2d_kernel *InstancePtr, u16 DeviceId) {
	XConv2d_kernel_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XConv2d_kernel_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XConv2d_kernel_CfgInitialize(InstancePtr, ConfigPtr);
}
#endif

#endif

