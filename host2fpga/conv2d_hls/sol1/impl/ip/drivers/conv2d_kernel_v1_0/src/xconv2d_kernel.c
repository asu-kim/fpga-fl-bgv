// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
// Tool Version Limit: 2024.05
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
// 
// ==============================================================
/***************************** Include Files *********************************/
#include "xconv2d_kernel.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XConv2d_kernel_CfgInitialize(XConv2d_kernel *InstancePtr, XConv2d_kernel_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XConv2d_kernel_Start(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XConv2d_kernel_IsDone(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XConv2d_kernel_IsIdle(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XConv2d_kernel_IsReady(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XConv2d_kernel_Continue(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL) & 0x80;
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL, Data | 0x10);
}

void XConv2d_kernel_EnableAutoRestart(XConv2d_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XConv2d_kernel_DisableAutoRestart(XConv2d_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_AP_CTRL, 0);
}

void XConv2d_kernel_Set_enc_weights(XConv2d_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_WEIGHTS_DATA, (u32)(Data));
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_WEIGHTS_DATA + 4, (u32)(Data >> 32));
}

u64 XConv2d_kernel_Get_enc_weights(XConv2d_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_WEIGHTS_DATA);
    Data += (u64)XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_WEIGHTS_DATA + 4) << 32;
    return Data;
}

void XConv2d_kernel_Set_enc_bias(XConv2d_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_BIAS_DATA, (u32)(Data));
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_BIAS_DATA + 4, (u32)(Data >> 32));
}

u64 XConv2d_kernel_Get_enc_bias(XConv2d_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_BIAS_DATA);
    Data += (u64)XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_BIAS_DATA + 4) << 32;
    return Data;
}

void XConv2d_kernel_Set_enc_input(XConv2d_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_INPUT_DATA, (u32)(Data));
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_INPUT_DATA + 4, (u32)(Data >> 32));
}

u64 XConv2d_kernel_Get_enc_input(XConv2d_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_INPUT_DATA);
    Data += (u64)XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_INPUT_DATA + 4) << 32;
    return Data;
}

void XConv2d_kernel_Set_enc_output(XConv2d_kernel *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_OUTPUT_DATA, (u32)(Data));
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_OUTPUT_DATA + 4, (u32)(Data >> 32));
}

u64 XConv2d_kernel_Get_enc_output(XConv2d_kernel *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_OUTPUT_DATA);
    Data += (u64)XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ENC_OUTPUT_DATA + 4) << 32;
    return Data;
}

void XConv2d_kernel_Set_rows(XConv2d_kernel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ROWS_DATA, Data);
}

u32 XConv2d_kernel_Get_rows(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ROWS_DATA);
    return Data;
}

void XConv2d_kernel_Set_cols(XConv2d_kernel *InstancePtr, u32 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_COLS_DATA, Data);
}

u32 XConv2d_kernel_Get_cols(XConv2d_kernel *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_COLS_DATA);
    return Data;
}

void XConv2d_kernel_InterruptGlobalEnable(XConv2d_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_GIE, 1);
}

void XConv2d_kernel_InterruptGlobalDisable(XConv2d_kernel *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_GIE, 0);
}

void XConv2d_kernel_InterruptEnable(XConv2d_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_IER);
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_IER, Register | Mask);
}

void XConv2d_kernel_InterruptDisable(XConv2d_kernel *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_IER);
    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_IER, Register & (~Mask));
}

void XConv2d_kernel_InterruptClear(XConv2d_kernel *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XConv2d_kernel_WriteReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ISR, Mask);
}

u32 XConv2d_kernel_InterruptGetEnabled(XConv2d_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_IER);
}

u32 XConv2d_kernel_InterruptGetStatus(XConv2d_kernel *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XConv2d_kernel_ReadReg(InstancePtr->Control_BaseAddress, XCONV2D_KERNEL_CONTROL_ADDR_ISR);
}

