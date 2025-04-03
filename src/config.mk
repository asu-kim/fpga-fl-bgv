TARGET := hw
PLATFORM := xilinx_u55c_gen3x16_xdma_3_202210_1
KERNEL := adder

XF_PROJ_ROOT ?= $(shell pwd)
TEMP_DIR := build
BUILD_DIR := $(TEMP_DIR)/$(TARGET)

include ./utils.mk
