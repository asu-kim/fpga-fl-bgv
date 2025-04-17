set moduleName conv2d_kernel
set isTopModule 1
set isCombinational 0
set isDatapathOnly 0
set isPipelined 0
set pipeline_type none
set FunctionProtocol ap_ctrl_chain
set isOneStateSeq 0
set ProfileFlag 0
set StallSigGenFlag 0
set isEnableWaveformDebug 1
set hasInterrupt 0
set DLRegFirstOffset 0
set DLRegItemOffset 0
set C_modelName {conv2d_kernel}
set C_modelType { void 0 }
set ap_memory_interface_dict [dict create]
set C_modelArgList {
	{ HBM0 int 64 regular {axi_master 0}  }
	{ HBM1 int 32 unused {axi_master 0}  }
	{ HBM2 int 64 unused {axi_master 0}  }
	{ enc_weights int 64 regular {axi_slave 0}  }
	{ enc_bias int 64 unused {axi_slave 0}  }
	{ enc_input int 64 unused {axi_slave 0}  }
	{ enc_output int 64 unused {axi_slave 0}  }
	{ rows int 32 unused {axi_slave 0}  }
	{ cols int 32 unused {axi_slave 0}  }
}
set hasAXIMCache 0
set hasAXIML2Cache 0
set AXIMCacheInstDict [dict create]
set C_modelArgMapList {[ 
	{ "Name" : "HBM0", "interface" : "axi_master", "bitwidth" : 64, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "enc_weights","offset": { "type": "dynamic","port_name": "enc_weights","bundle": "control"},"direction": "READONLY"},{"cName": "enc_bias","offset": { "type": "dynamic","port_name": "enc_bias","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "HBM1", "interface" : "axi_master", "bitwidth" : 32, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "enc_input","offset": { "type": "dynamic","port_name": "enc_input","bundle": "control"},"direction": "READONLY"}]}]} , 
 	{ "Name" : "HBM2", "interface" : "axi_master", "bitwidth" : 64, "direction" : "READONLY", "bitSlice":[ {"cElement": [{"cName": "enc_output","offset": { "type": "dynamic","port_name": "enc_output","bundle": "control"},"direction": "WRITEONLY"}]}]} , 
 	{ "Name" : "enc_weights", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":16}, "offset_end" : {"in":27}} , 
 	{ "Name" : "enc_bias", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":28}, "offset_end" : {"in":39}} , 
 	{ "Name" : "enc_input", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":40}, "offset_end" : {"in":51}} , 
 	{ "Name" : "enc_output", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 64, "direction" : "READONLY", "offset" : {"in":52}, "offset_end" : {"in":63}} , 
 	{ "Name" : "rows", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":64}, "offset_end" : {"in":71}} , 
 	{ "Name" : "cols", "interface" : "axi_slave", "bundle":"control","type":"ap_none","bitwidth" : 32, "direction" : "READONLY", "offset" : {"in":72}, "offset_end" : {"in":79}} ]}
# RTL Port declarations: 
set portNum 155
set portList { 
	{ ap_clk sc_in sc_logic 1 clock -1 } 
	{ ap_rst_n sc_in sc_logic 1 reset -1 active_low_sync } 
	{ m_axi_HBM0_AWVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_HBM0_AWREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_HBM0_AWADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_HBM0_AWID sc_out sc_lv 1 signal 0 } 
	{ m_axi_HBM0_AWLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_HBM0_AWSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_HBM0_AWBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_HBM0_AWLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_HBM0_AWCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_HBM0_AWPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_HBM0_AWQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_HBM0_AWREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_HBM0_AWUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_HBM0_WVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_HBM0_WREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_HBM0_WDATA sc_out sc_lv 64 signal 0 } 
	{ m_axi_HBM0_WSTRB sc_out sc_lv 8 signal 0 } 
	{ m_axi_HBM0_WLAST sc_out sc_logic 1 signal 0 } 
	{ m_axi_HBM0_WID sc_out sc_lv 1 signal 0 } 
	{ m_axi_HBM0_WUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_HBM0_ARVALID sc_out sc_logic 1 signal 0 } 
	{ m_axi_HBM0_ARREADY sc_in sc_logic 1 signal 0 } 
	{ m_axi_HBM0_ARADDR sc_out sc_lv 64 signal 0 } 
	{ m_axi_HBM0_ARID sc_out sc_lv 1 signal 0 } 
	{ m_axi_HBM0_ARLEN sc_out sc_lv 8 signal 0 } 
	{ m_axi_HBM0_ARSIZE sc_out sc_lv 3 signal 0 } 
	{ m_axi_HBM0_ARBURST sc_out sc_lv 2 signal 0 } 
	{ m_axi_HBM0_ARLOCK sc_out sc_lv 2 signal 0 } 
	{ m_axi_HBM0_ARCACHE sc_out sc_lv 4 signal 0 } 
	{ m_axi_HBM0_ARPROT sc_out sc_lv 3 signal 0 } 
	{ m_axi_HBM0_ARQOS sc_out sc_lv 4 signal 0 } 
	{ m_axi_HBM0_ARREGION sc_out sc_lv 4 signal 0 } 
	{ m_axi_HBM0_ARUSER sc_out sc_lv 1 signal 0 } 
	{ m_axi_HBM0_RVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_HBM0_RREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_HBM0_RDATA sc_in sc_lv 64 signal 0 } 
	{ m_axi_HBM0_RLAST sc_in sc_logic 1 signal 0 } 
	{ m_axi_HBM0_RID sc_in sc_lv 1 signal 0 } 
	{ m_axi_HBM0_RUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_HBM0_RRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_HBM0_BVALID sc_in sc_logic 1 signal 0 } 
	{ m_axi_HBM0_BREADY sc_out sc_logic 1 signal 0 } 
	{ m_axi_HBM0_BRESP sc_in sc_lv 2 signal 0 } 
	{ m_axi_HBM0_BID sc_in sc_lv 1 signal 0 } 
	{ m_axi_HBM0_BUSER sc_in sc_lv 1 signal 0 } 
	{ m_axi_HBM1_AWVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_HBM1_AWREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_HBM1_AWADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_HBM1_AWID sc_out sc_lv 1 signal 1 } 
	{ m_axi_HBM1_AWLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_HBM1_AWSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_HBM1_AWBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_HBM1_AWLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_HBM1_AWCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_AWPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_HBM1_AWQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_AWREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_AWUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_HBM1_WVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_HBM1_WREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_HBM1_WDATA sc_out sc_lv 32 signal 1 } 
	{ m_axi_HBM1_WSTRB sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_WLAST sc_out sc_logic 1 signal 1 } 
	{ m_axi_HBM1_WID sc_out sc_lv 1 signal 1 } 
	{ m_axi_HBM1_WUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_HBM1_ARVALID sc_out sc_logic 1 signal 1 } 
	{ m_axi_HBM1_ARREADY sc_in sc_logic 1 signal 1 } 
	{ m_axi_HBM1_ARADDR sc_out sc_lv 64 signal 1 } 
	{ m_axi_HBM1_ARID sc_out sc_lv 1 signal 1 } 
	{ m_axi_HBM1_ARLEN sc_out sc_lv 8 signal 1 } 
	{ m_axi_HBM1_ARSIZE sc_out sc_lv 3 signal 1 } 
	{ m_axi_HBM1_ARBURST sc_out sc_lv 2 signal 1 } 
	{ m_axi_HBM1_ARLOCK sc_out sc_lv 2 signal 1 } 
	{ m_axi_HBM1_ARCACHE sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_ARPROT sc_out sc_lv 3 signal 1 } 
	{ m_axi_HBM1_ARQOS sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_ARREGION sc_out sc_lv 4 signal 1 } 
	{ m_axi_HBM1_ARUSER sc_out sc_lv 1 signal 1 } 
	{ m_axi_HBM1_RVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_HBM1_RREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_HBM1_RDATA sc_in sc_lv 32 signal 1 } 
	{ m_axi_HBM1_RLAST sc_in sc_logic 1 signal 1 } 
	{ m_axi_HBM1_RID sc_in sc_lv 1 signal 1 } 
	{ m_axi_HBM1_RUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_HBM1_RRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_HBM1_BVALID sc_in sc_logic 1 signal 1 } 
	{ m_axi_HBM1_BREADY sc_out sc_logic 1 signal 1 } 
	{ m_axi_HBM1_BRESP sc_in sc_lv 2 signal 1 } 
	{ m_axi_HBM1_BID sc_in sc_lv 1 signal 1 } 
	{ m_axi_HBM1_BUSER sc_in sc_lv 1 signal 1 } 
	{ m_axi_HBM2_AWVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_HBM2_AWREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_HBM2_AWADDR sc_out sc_lv 64 signal 2 } 
	{ m_axi_HBM2_AWID sc_out sc_lv 1 signal 2 } 
	{ m_axi_HBM2_AWLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_HBM2_AWSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_HBM2_AWBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_HBM2_AWLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_HBM2_AWCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_HBM2_AWPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_HBM2_AWQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_HBM2_AWREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_HBM2_AWUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_HBM2_WVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_HBM2_WREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_HBM2_WDATA sc_out sc_lv 64 signal 2 } 
	{ m_axi_HBM2_WSTRB sc_out sc_lv 8 signal 2 } 
	{ m_axi_HBM2_WLAST sc_out sc_logic 1 signal 2 } 
	{ m_axi_HBM2_WID sc_out sc_lv 1 signal 2 } 
	{ m_axi_HBM2_WUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_HBM2_ARVALID sc_out sc_logic 1 signal 2 } 
	{ m_axi_HBM2_ARREADY sc_in sc_logic 1 signal 2 } 
	{ m_axi_HBM2_ARADDR sc_out sc_lv 64 signal 2 } 
	{ m_axi_HBM2_ARID sc_out sc_lv 1 signal 2 } 
	{ m_axi_HBM2_ARLEN sc_out sc_lv 8 signal 2 } 
	{ m_axi_HBM2_ARSIZE sc_out sc_lv 3 signal 2 } 
	{ m_axi_HBM2_ARBURST sc_out sc_lv 2 signal 2 } 
	{ m_axi_HBM2_ARLOCK sc_out sc_lv 2 signal 2 } 
	{ m_axi_HBM2_ARCACHE sc_out sc_lv 4 signal 2 } 
	{ m_axi_HBM2_ARPROT sc_out sc_lv 3 signal 2 } 
	{ m_axi_HBM2_ARQOS sc_out sc_lv 4 signal 2 } 
	{ m_axi_HBM2_ARREGION sc_out sc_lv 4 signal 2 } 
	{ m_axi_HBM2_ARUSER sc_out sc_lv 1 signal 2 } 
	{ m_axi_HBM2_RVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_HBM2_RREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_HBM2_RDATA sc_in sc_lv 64 signal 2 } 
	{ m_axi_HBM2_RLAST sc_in sc_logic 1 signal 2 } 
	{ m_axi_HBM2_RID sc_in sc_lv 1 signal 2 } 
	{ m_axi_HBM2_RUSER sc_in sc_lv 1 signal 2 } 
	{ m_axi_HBM2_RRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_HBM2_BVALID sc_in sc_logic 1 signal 2 } 
	{ m_axi_HBM2_BREADY sc_out sc_logic 1 signal 2 } 
	{ m_axi_HBM2_BRESP sc_in sc_lv 2 signal 2 } 
	{ m_axi_HBM2_BID sc_in sc_lv 1 signal 2 } 
	{ m_axi_HBM2_BUSER sc_in sc_lv 1 signal 2 } 
	{ s_axi_control_AWVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_AWREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_AWADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_WVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_WREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_WDATA sc_in sc_lv 32 signal -1 } 
	{ s_axi_control_WSTRB sc_in sc_lv 4 signal -1 } 
	{ s_axi_control_ARVALID sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_ARREADY sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_ARADDR sc_in sc_lv 7 signal -1 } 
	{ s_axi_control_RVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_RREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_RDATA sc_out sc_lv 32 signal -1 } 
	{ s_axi_control_RRESP sc_out sc_lv 2 signal -1 } 
	{ s_axi_control_BVALID sc_out sc_logic 1 signal -1 } 
	{ s_axi_control_BREADY sc_in sc_logic 1 signal -1 } 
	{ s_axi_control_BRESP sc_out sc_lv 2 signal -1 } 
	{ interrupt sc_out sc_logic 1 signal -1 } 
}
set NewPortList {[ 
	{ "name": "s_axi_control_AWADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "AWADDR" },"address":[{"name":"conv2d_kernel","role":"start","value":"0","valid_bit":"0"},{"name":"conv2d_kernel","role":"continue","value":"0","valid_bit":"4"},{"name":"conv2d_kernel","role":"auto_start","value":"0","valid_bit":"7"},{"name":"enc_weights","role":"data","value":"16"},{"name":"enc_bias","role":"data","value":"28"},{"name":"enc_input","role":"data","value":"40"},{"name":"enc_output","role":"data","value":"52"},{"name":"rows","role":"data","value":"64"},{"name":"cols","role":"data","value":"72"}] },
	{ "name": "s_axi_control_AWVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWVALID" } },
	{ "name": "s_axi_control_AWREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "AWREADY" } },
	{ "name": "s_axi_control_WVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WVALID" } },
	{ "name": "s_axi_control_WREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "WREADY" } },
	{ "name": "s_axi_control_WDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "WDATA" } },
	{ "name": "s_axi_control_WSTRB", "direction": "in", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "control", "role": "WSTRB" } },
	{ "name": "s_axi_control_ARADDR", "direction": "in", "datatype": "sc_lv", "bitwidth":7, "type": "signal", "bundle":{"name": "control", "role": "ARADDR" },"address":[{"name":"conv2d_kernel","role":"start","value":"0","valid_bit":"0"},{"name":"conv2d_kernel","role":"done","value":"0","valid_bit":"1"},{"name":"conv2d_kernel","role":"idle","value":"0","valid_bit":"2"},{"name":"conv2d_kernel","role":"ready","value":"0","valid_bit":"3"},{"name":"conv2d_kernel","role":"auto_start","value":"0","valid_bit":"7"}] },
	{ "name": "s_axi_control_ARVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARVALID" } },
	{ "name": "s_axi_control_ARREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "ARREADY" } },
	{ "name": "s_axi_control_RVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RVALID" } },
	{ "name": "s_axi_control_RREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "RREADY" } },
	{ "name": "s_axi_control_RDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "control", "role": "RDATA" } },
	{ "name": "s_axi_control_RRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "RRESP" } },
	{ "name": "s_axi_control_BVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BVALID" } },
	{ "name": "s_axi_control_BREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "BREADY" } },
	{ "name": "s_axi_control_BRESP", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "control", "role": "BRESP" } },
	{ "name": "interrupt", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "control", "role": "interrupt" } }, 
 	{ "name": "ap_clk", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "clock", "bundle":{"name": "ap_clk", "role": "default" }} , 
 	{ "name": "ap_rst_n", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "reset", "bundle":{"name": "ap_rst_n", "role": "default" }} , 
 	{ "name": "m_axi_HBM0_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "AWVALID" }} , 
 	{ "name": "m_axi_HBM0_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "AWREADY" }} , 
 	{ "name": "m_axi_HBM0_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM0", "role": "AWADDR" }} , 
 	{ "name": "m_axi_HBM0_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "AWID" }} , 
 	{ "name": "m_axi_HBM0_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM0", "role": "AWLEN" }} , 
 	{ "name": "m_axi_HBM0_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM0", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_HBM0_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM0", "role": "AWBURST" }} , 
 	{ "name": "m_axi_HBM0_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM0", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_HBM0_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM0", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_HBM0_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM0", "role": "AWPROT" }} , 
 	{ "name": "m_axi_HBM0_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM0", "role": "AWQOS" }} , 
 	{ "name": "m_axi_HBM0_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM0", "role": "AWREGION" }} , 
 	{ "name": "m_axi_HBM0_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "AWUSER" }} , 
 	{ "name": "m_axi_HBM0_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "WVALID" }} , 
 	{ "name": "m_axi_HBM0_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "WREADY" }} , 
 	{ "name": "m_axi_HBM0_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM0", "role": "WDATA" }} , 
 	{ "name": "m_axi_HBM0_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM0", "role": "WSTRB" }} , 
 	{ "name": "m_axi_HBM0_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "WLAST" }} , 
 	{ "name": "m_axi_HBM0_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "WID" }} , 
 	{ "name": "m_axi_HBM0_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "WUSER" }} , 
 	{ "name": "m_axi_HBM0_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "ARVALID" }} , 
 	{ "name": "m_axi_HBM0_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "ARREADY" }} , 
 	{ "name": "m_axi_HBM0_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM0", "role": "ARADDR" }} , 
 	{ "name": "m_axi_HBM0_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "ARID" }} , 
 	{ "name": "m_axi_HBM0_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM0", "role": "ARLEN" }} , 
 	{ "name": "m_axi_HBM0_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM0", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_HBM0_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM0", "role": "ARBURST" }} , 
 	{ "name": "m_axi_HBM0_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM0", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_HBM0_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM0", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_HBM0_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM0", "role": "ARPROT" }} , 
 	{ "name": "m_axi_HBM0_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM0", "role": "ARQOS" }} , 
 	{ "name": "m_axi_HBM0_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM0", "role": "ARREGION" }} , 
 	{ "name": "m_axi_HBM0_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "ARUSER" }} , 
 	{ "name": "m_axi_HBM0_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "RVALID" }} , 
 	{ "name": "m_axi_HBM0_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "RREADY" }} , 
 	{ "name": "m_axi_HBM0_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM0", "role": "RDATA" }} , 
 	{ "name": "m_axi_HBM0_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "RLAST" }} , 
 	{ "name": "m_axi_HBM0_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "RID" }} , 
 	{ "name": "m_axi_HBM0_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "RUSER" }} , 
 	{ "name": "m_axi_HBM0_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM0", "role": "RRESP" }} , 
 	{ "name": "m_axi_HBM0_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "BVALID" }} , 
 	{ "name": "m_axi_HBM0_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "BREADY" }} , 
 	{ "name": "m_axi_HBM0_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM0", "role": "BRESP" }} , 
 	{ "name": "m_axi_HBM0_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "BID" }} , 
 	{ "name": "m_axi_HBM0_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM0", "role": "BUSER" }} , 
 	{ "name": "m_axi_HBM1_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "AWVALID" }} , 
 	{ "name": "m_axi_HBM1_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "AWREADY" }} , 
 	{ "name": "m_axi_HBM1_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM1", "role": "AWADDR" }} , 
 	{ "name": "m_axi_HBM1_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "AWID" }} , 
 	{ "name": "m_axi_HBM1_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM1", "role": "AWLEN" }} , 
 	{ "name": "m_axi_HBM1_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM1", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_HBM1_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM1", "role": "AWBURST" }} , 
 	{ "name": "m_axi_HBM1_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM1", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_HBM1_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_HBM1_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM1", "role": "AWPROT" }} , 
 	{ "name": "m_axi_HBM1_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "AWQOS" }} , 
 	{ "name": "m_axi_HBM1_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "AWREGION" }} , 
 	{ "name": "m_axi_HBM1_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "AWUSER" }} , 
 	{ "name": "m_axi_HBM1_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "WVALID" }} , 
 	{ "name": "m_axi_HBM1_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "WREADY" }} , 
 	{ "name": "m_axi_HBM1_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "HBM1", "role": "WDATA" }} , 
 	{ "name": "m_axi_HBM1_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "WSTRB" }} , 
 	{ "name": "m_axi_HBM1_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "WLAST" }} , 
 	{ "name": "m_axi_HBM1_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "WID" }} , 
 	{ "name": "m_axi_HBM1_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "WUSER" }} , 
 	{ "name": "m_axi_HBM1_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "ARVALID" }} , 
 	{ "name": "m_axi_HBM1_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "ARREADY" }} , 
 	{ "name": "m_axi_HBM1_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM1", "role": "ARADDR" }} , 
 	{ "name": "m_axi_HBM1_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "ARID" }} , 
 	{ "name": "m_axi_HBM1_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM1", "role": "ARLEN" }} , 
 	{ "name": "m_axi_HBM1_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM1", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_HBM1_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM1", "role": "ARBURST" }} , 
 	{ "name": "m_axi_HBM1_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM1", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_HBM1_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_HBM1_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM1", "role": "ARPROT" }} , 
 	{ "name": "m_axi_HBM1_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "ARQOS" }} , 
 	{ "name": "m_axi_HBM1_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM1", "role": "ARREGION" }} , 
 	{ "name": "m_axi_HBM1_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "ARUSER" }} , 
 	{ "name": "m_axi_HBM1_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "RVALID" }} , 
 	{ "name": "m_axi_HBM1_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "RREADY" }} , 
 	{ "name": "m_axi_HBM1_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":32, "type": "signal", "bundle":{"name": "HBM1", "role": "RDATA" }} , 
 	{ "name": "m_axi_HBM1_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "RLAST" }} , 
 	{ "name": "m_axi_HBM1_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "RID" }} , 
 	{ "name": "m_axi_HBM1_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "RUSER" }} , 
 	{ "name": "m_axi_HBM1_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM1", "role": "RRESP" }} , 
 	{ "name": "m_axi_HBM1_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "BVALID" }} , 
 	{ "name": "m_axi_HBM1_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "BREADY" }} , 
 	{ "name": "m_axi_HBM1_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM1", "role": "BRESP" }} , 
 	{ "name": "m_axi_HBM1_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "BID" }} , 
 	{ "name": "m_axi_HBM1_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM1", "role": "BUSER" }} , 
 	{ "name": "m_axi_HBM2_AWVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "AWVALID" }} , 
 	{ "name": "m_axi_HBM2_AWREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "AWREADY" }} , 
 	{ "name": "m_axi_HBM2_AWADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM2", "role": "AWADDR" }} , 
 	{ "name": "m_axi_HBM2_AWID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "AWID" }} , 
 	{ "name": "m_axi_HBM2_AWLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM2", "role": "AWLEN" }} , 
 	{ "name": "m_axi_HBM2_AWSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM2", "role": "AWSIZE" }} , 
 	{ "name": "m_axi_HBM2_AWBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM2", "role": "AWBURST" }} , 
 	{ "name": "m_axi_HBM2_AWLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM2", "role": "AWLOCK" }} , 
 	{ "name": "m_axi_HBM2_AWCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM2", "role": "AWCACHE" }} , 
 	{ "name": "m_axi_HBM2_AWPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM2", "role": "AWPROT" }} , 
 	{ "name": "m_axi_HBM2_AWQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM2", "role": "AWQOS" }} , 
 	{ "name": "m_axi_HBM2_AWREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM2", "role": "AWREGION" }} , 
 	{ "name": "m_axi_HBM2_AWUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "AWUSER" }} , 
 	{ "name": "m_axi_HBM2_WVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "WVALID" }} , 
 	{ "name": "m_axi_HBM2_WREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "WREADY" }} , 
 	{ "name": "m_axi_HBM2_WDATA", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM2", "role": "WDATA" }} , 
 	{ "name": "m_axi_HBM2_WSTRB", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM2", "role": "WSTRB" }} , 
 	{ "name": "m_axi_HBM2_WLAST", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "WLAST" }} , 
 	{ "name": "m_axi_HBM2_WID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "WID" }} , 
 	{ "name": "m_axi_HBM2_WUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "WUSER" }} , 
 	{ "name": "m_axi_HBM2_ARVALID", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "ARVALID" }} , 
 	{ "name": "m_axi_HBM2_ARREADY", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "ARREADY" }} , 
 	{ "name": "m_axi_HBM2_ARADDR", "direction": "out", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM2", "role": "ARADDR" }} , 
 	{ "name": "m_axi_HBM2_ARID", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "ARID" }} , 
 	{ "name": "m_axi_HBM2_ARLEN", "direction": "out", "datatype": "sc_lv", "bitwidth":8, "type": "signal", "bundle":{"name": "HBM2", "role": "ARLEN" }} , 
 	{ "name": "m_axi_HBM2_ARSIZE", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM2", "role": "ARSIZE" }} , 
 	{ "name": "m_axi_HBM2_ARBURST", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM2", "role": "ARBURST" }} , 
 	{ "name": "m_axi_HBM2_ARLOCK", "direction": "out", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM2", "role": "ARLOCK" }} , 
 	{ "name": "m_axi_HBM2_ARCACHE", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM2", "role": "ARCACHE" }} , 
 	{ "name": "m_axi_HBM2_ARPROT", "direction": "out", "datatype": "sc_lv", "bitwidth":3, "type": "signal", "bundle":{"name": "HBM2", "role": "ARPROT" }} , 
 	{ "name": "m_axi_HBM2_ARQOS", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM2", "role": "ARQOS" }} , 
 	{ "name": "m_axi_HBM2_ARREGION", "direction": "out", "datatype": "sc_lv", "bitwidth":4, "type": "signal", "bundle":{"name": "HBM2", "role": "ARREGION" }} , 
 	{ "name": "m_axi_HBM2_ARUSER", "direction": "out", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "ARUSER" }} , 
 	{ "name": "m_axi_HBM2_RVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "RVALID" }} , 
 	{ "name": "m_axi_HBM2_RREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "RREADY" }} , 
 	{ "name": "m_axi_HBM2_RDATA", "direction": "in", "datatype": "sc_lv", "bitwidth":64, "type": "signal", "bundle":{"name": "HBM2", "role": "RDATA" }} , 
 	{ "name": "m_axi_HBM2_RLAST", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "RLAST" }} , 
 	{ "name": "m_axi_HBM2_RID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "RID" }} , 
 	{ "name": "m_axi_HBM2_RUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "RUSER" }} , 
 	{ "name": "m_axi_HBM2_RRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM2", "role": "RRESP" }} , 
 	{ "name": "m_axi_HBM2_BVALID", "direction": "in", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "BVALID" }} , 
 	{ "name": "m_axi_HBM2_BREADY", "direction": "out", "datatype": "sc_logic", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "BREADY" }} , 
 	{ "name": "m_axi_HBM2_BRESP", "direction": "in", "datatype": "sc_lv", "bitwidth":2, "type": "signal", "bundle":{"name": "HBM2", "role": "BRESP" }} , 
 	{ "name": "m_axi_HBM2_BID", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "BID" }} , 
 	{ "name": "m_axi_HBM2_BUSER", "direction": "in", "datatype": "sc_lv", "bitwidth":1, "type": "signal", "bundle":{"name": "HBM2", "role": "BUSER" }}  ]}

set RtlHierarchyInfo {[
	{"ID" : "0", "Level" : "0", "Path" : "`AUTOTB_DUT_INST", "Parent" : "", "Child" : ["1", "2", "3"],
		"CDFG" : "conv2d_kernel",
		"Protocol" : "ap_ctrl_chain",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "1", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "HBM0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "HBM0_blk_n_AR", "Type" : "RtlSignal"}],
				"SubConnect" : [
					{"ID" : "1", "SubInstance" : "grp_conv2d_kernel_Pipeline_VITIS_LOOP_42_1_fu_99", "Port" : "HBM0", "Inst_start_state" : "73", "Inst_end_state" : "74"}]},
			{"Name" : "HBM1", "Type" : "MAXI", "Direction" : "I"},
			{"Name" : "HBM2", "Type" : "MAXI", "Direction" : "I"},
			{"Name" : "enc_weights", "Type" : "None", "Direction" : "I"},
			{"Name" : "enc_bias", "Type" : "None", "Direction" : "I"},
			{"Name" : "enc_input", "Type" : "None", "Direction" : "I"},
			{"Name" : "enc_output", "Type" : "None", "Direction" : "I"},
			{"Name" : "rows", "Type" : "None", "Direction" : "I"},
			{"Name" : "cols", "Type" : "None", "Direction" : "I"}]},
	{"ID" : "1", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.grp_conv2d_kernel_Pipeline_VITIS_LOOP_42_1_fu_99", "Parent" : "0",
		"CDFG" : "conv2d_kernel_Pipeline_VITIS_LOOP_42_1",
		"Protocol" : "ap_ctrl_hs",
		"ControlExist" : "1", "ap_start" : "1", "ap_ready" : "1", "ap_done" : "1", "ap_continue" : "0", "ap_idle" : "1", "real_start" : "0",
		"Pipeline" : "None", "UnalignedPipeline" : "0", "RewindPipeline" : "0", "ProcessNetwork" : "0",
		"II" : "0",
		"VariableLatency" : "1", "ExactLatency" : "-1", "EstimateLatencyMin" : "-1", "EstimateLatencyMax" : "-1",
		"Combinational" : "0",
		"Datapath" : "0",
		"ClockEnable" : "0",
		"HasSubDataflow" : "0",
		"InDataflowNetwork" : "0",
		"HasNonBlockingOperation" : "0",
		"IsBlackBox" : "0",
		"Port" : [
			{"Name" : "HBM0", "Type" : "MAXI", "Direction" : "I",
				"BlockSignal" : [
					{"Name" : "HBM0_blk_n_R", "Type" : "RtlSignal"}]},
			{"Name" : "sext_ln42", "Type" : "None", "Direction" : "I"}],
		"Loop" : [
			{"Name" : "VITIS_LOOP_42_1", "PipelineType" : "pipeline",
				"LoopDec" : {"FSMBitwidth" : "2", "FirstState" : "ap_ST_fsm_pp0_stage0", "FirstStateIter" : "ap_enable_reg_pp0_iter0", "FirstStateBlock" : "ap_block_pp0_stage0_subdone", "LastState" : "ap_ST_fsm_pp0_stage0", "LastStateIter" : "ap_enable_reg_pp0_iter1", "LastStateBlock" : "ap_block_pp0_stage0_subdone", "PreState" : ["ap_ST_fsm_state1"], "QuitState" : "ap_ST_fsm_pp0_stage0", "QuitStateIter" : "ap_enable_reg_pp0_iter1", "QuitStateBlock" : "ap_block_pp0_stage0_subdone", "PostState" : []}}]},
	{"ID" : "2", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.control_s_axi_U", "Parent" : "0"},
	{"ID" : "3", "Level" : "1", "Path" : "`AUTOTB_DUT_INST.HBM0_m_axi_U", "Parent" : "0"}]}


set ArgLastReadFirstWriteLatency {
	conv2d_kernel {
		HBM0 {Type I LastRead 2 FirstWrite -1}
		HBM1 {Type I LastRead -1 FirstWrite -1}
		HBM2 {Type I LastRead -1 FirstWrite -1}
		enc_weights {Type I LastRead 0 FirstWrite -1}
		enc_bias {Type I LastRead -1 FirstWrite -1}
		enc_input {Type I LastRead -1 FirstWrite -1}
		enc_output {Type I LastRead -1 FirstWrite -1}
		rows {Type I LastRead -1 FirstWrite -1}
		cols {Type I LastRead -1 FirstWrite -1}}
	conv2d_kernel_Pipeline_VITIS_LOOP_42_1 {
		HBM0 {Type I LastRead 2 FirstWrite -1}
		sext_ln42 {Type I LastRead 0 FirstWrite -1}}}

set hasDtUnsupportedChannel 0

set PerformanceInfo {[
	{"Name" : "Latency", "Min" : "-1", "Max" : "-1"}
	, {"Name" : "Interval", "Min" : "0", "Max" : "0"}
]}

set PipelineEnableSignalInfo {[
]}

set Spec2ImplPortList { 
	HBM0 { m_axi {  { m_axi_HBM0_AWVALID VALID 1 1 }  { m_axi_HBM0_AWREADY READY 0 1 }  { m_axi_HBM0_AWADDR ADDR 1 64 }  { m_axi_HBM0_AWID ID 1 1 }  { m_axi_HBM0_AWLEN SIZE 1 8 }  { m_axi_HBM0_AWSIZE BURST 1 3 }  { m_axi_HBM0_AWBURST LOCK 1 2 }  { m_axi_HBM0_AWLOCK CACHE 1 2 }  { m_axi_HBM0_AWCACHE PROT 1 4 }  { m_axi_HBM0_AWPROT QOS 1 3 }  { m_axi_HBM0_AWQOS REGION 1 4 }  { m_axi_HBM0_AWREGION USER 1 4 }  { m_axi_HBM0_AWUSER DATA 1 1 }  { m_axi_HBM0_WVALID VALID 1 1 }  { m_axi_HBM0_WREADY READY 0 1 }  { m_axi_HBM0_WDATA FIFONUM 1 64 }  { m_axi_HBM0_WSTRB STRB 1 8 }  { m_axi_HBM0_WLAST LAST 1 1 }  { m_axi_HBM0_WID ID 1 1 }  { m_axi_HBM0_WUSER DATA 1 1 }  { m_axi_HBM0_ARVALID VALID 1 1 }  { m_axi_HBM0_ARREADY READY 0 1 }  { m_axi_HBM0_ARADDR ADDR 1 64 }  { m_axi_HBM0_ARID ID 1 1 }  { m_axi_HBM0_ARLEN SIZE 1 8 }  { m_axi_HBM0_ARSIZE BURST 1 3 }  { m_axi_HBM0_ARBURST LOCK 1 2 }  { m_axi_HBM0_ARLOCK CACHE 1 2 }  { m_axi_HBM0_ARCACHE PROT 1 4 }  { m_axi_HBM0_ARPROT QOS 1 3 }  { m_axi_HBM0_ARQOS REGION 1 4 }  { m_axi_HBM0_ARREGION USER 1 4 }  { m_axi_HBM0_ARUSER DATA 1 1 }  { m_axi_HBM0_RVALID VALID 0 1 }  { m_axi_HBM0_RREADY READY 1 1 }  { m_axi_HBM0_RDATA FIFONUM 0 64 }  { m_axi_HBM0_RLAST LAST 0 1 }  { m_axi_HBM0_RID ID 0 1 }  { m_axi_HBM0_RUSER DATA 0 1 }  { m_axi_HBM0_RRESP RESP 0 2 }  { m_axi_HBM0_BVALID VALID 0 1 }  { m_axi_HBM0_BREADY READY 1 1 }  { m_axi_HBM0_BRESP RESP 0 2 }  { m_axi_HBM0_BID ID 0 1 }  { m_axi_HBM0_BUSER DATA 0 1 } } }
	HBM1 { m_axi {  { m_axi_HBM1_AWVALID VALID 1 1 }  { m_axi_HBM1_AWREADY READY 0 1 }  { m_axi_HBM1_AWADDR ADDR 1 64 }  { m_axi_HBM1_AWID ID 1 1 }  { m_axi_HBM1_AWLEN SIZE 1 8 }  { m_axi_HBM1_AWSIZE BURST 1 3 }  { m_axi_HBM1_AWBURST LOCK 1 2 }  { m_axi_HBM1_AWLOCK CACHE 1 2 }  { m_axi_HBM1_AWCACHE PROT 1 4 }  { m_axi_HBM1_AWPROT QOS 1 3 }  { m_axi_HBM1_AWQOS REGION 1 4 }  { m_axi_HBM1_AWREGION USER 1 4 }  { m_axi_HBM1_AWUSER DATA 1 1 }  { m_axi_HBM1_WVALID VALID 1 1 }  { m_axi_HBM1_WREADY READY 0 1 }  { m_axi_HBM1_WDATA FIFONUM 1 32 }  { m_axi_HBM1_WSTRB STRB 1 4 }  { m_axi_HBM1_WLAST LAST 1 1 }  { m_axi_HBM1_WID ID 1 1 }  { m_axi_HBM1_WUSER DATA 1 1 }  { m_axi_HBM1_ARVALID VALID 1 1 }  { m_axi_HBM1_ARREADY READY 0 1 }  { m_axi_HBM1_ARADDR ADDR 1 64 }  { m_axi_HBM1_ARID ID 1 1 }  { m_axi_HBM1_ARLEN SIZE 1 8 }  { m_axi_HBM1_ARSIZE BURST 1 3 }  { m_axi_HBM1_ARBURST LOCK 1 2 }  { m_axi_HBM1_ARLOCK CACHE 1 2 }  { m_axi_HBM1_ARCACHE PROT 1 4 }  { m_axi_HBM1_ARPROT QOS 1 3 }  { m_axi_HBM1_ARQOS REGION 1 4 }  { m_axi_HBM1_ARREGION USER 1 4 }  { m_axi_HBM1_ARUSER DATA 1 1 }  { m_axi_HBM1_RVALID VALID 0 1 }  { m_axi_HBM1_RREADY READY 1 1 }  { m_axi_HBM1_RDATA FIFONUM 0 32 }  { m_axi_HBM1_RLAST LAST 0 1 }  { m_axi_HBM1_RID ID 0 1 }  { m_axi_HBM1_RUSER DATA 0 1 }  { m_axi_HBM1_RRESP RESP 0 2 }  { m_axi_HBM1_BVALID VALID 0 1 }  { m_axi_HBM1_BREADY READY 1 1 }  { m_axi_HBM1_BRESP RESP 0 2 }  { m_axi_HBM1_BID ID 0 1 }  { m_axi_HBM1_BUSER DATA 0 1 } } }
	HBM2 { m_axi {  { m_axi_HBM2_AWVALID VALID 1 1 }  { m_axi_HBM2_AWREADY READY 0 1 }  { m_axi_HBM2_AWADDR ADDR 1 64 }  { m_axi_HBM2_AWID ID 1 1 }  { m_axi_HBM2_AWLEN SIZE 1 8 }  { m_axi_HBM2_AWSIZE BURST 1 3 }  { m_axi_HBM2_AWBURST LOCK 1 2 }  { m_axi_HBM2_AWLOCK CACHE 1 2 }  { m_axi_HBM2_AWCACHE PROT 1 4 }  { m_axi_HBM2_AWPROT QOS 1 3 }  { m_axi_HBM2_AWQOS REGION 1 4 }  { m_axi_HBM2_AWREGION USER 1 4 }  { m_axi_HBM2_AWUSER DATA 1 1 }  { m_axi_HBM2_WVALID VALID 1 1 }  { m_axi_HBM2_WREADY READY 0 1 }  { m_axi_HBM2_WDATA FIFONUM 1 64 }  { m_axi_HBM2_WSTRB STRB 1 8 }  { m_axi_HBM2_WLAST LAST 1 1 }  { m_axi_HBM2_WID ID 1 1 }  { m_axi_HBM2_WUSER DATA 1 1 }  { m_axi_HBM2_ARVALID VALID 1 1 }  { m_axi_HBM2_ARREADY READY 0 1 }  { m_axi_HBM2_ARADDR ADDR 1 64 }  { m_axi_HBM2_ARID ID 1 1 }  { m_axi_HBM2_ARLEN SIZE 1 8 }  { m_axi_HBM2_ARSIZE BURST 1 3 }  { m_axi_HBM2_ARBURST LOCK 1 2 }  { m_axi_HBM2_ARLOCK CACHE 1 2 }  { m_axi_HBM2_ARCACHE PROT 1 4 }  { m_axi_HBM2_ARPROT QOS 1 3 }  { m_axi_HBM2_ARQOS REGION 1 4 }  { m_axi_HBM2_ARREGION USER 1 4 }  { m_axi_HBM2_ARUSER DATA 1 1 }  { m_axi_HBM2_RVALID VALID 0 1 }  { m_axi_HBM2_RREADY READY 1 1 }  { m_axi_HBM2_RDATA FIFONUM 0 64 }  { m_axi_HBM2_RLAST LAST 0 1 }  { m_axi_HBM2_RID ID 0 1 }  { m_axi_HBM2_RUSER DATA 0 1 }  { m_axi_HBM2_RRESP RESP 0 2 }  { m_axi_HBM2_BVALID VALID 0 1 }  { m_axi_HBM2_BREADY READY 1 1 }  { m_axi_HBM2_BRESP RESP 0 2 }  { m_axi_HBM2_BID ID 0 1 }  { m_axi_HBM2_BUSER DATA 0 1 } } }
}

set maxi_interface_dict [dict create]
dict set maxi_interface_dict HBM0 { CHANNEL_NUM 0 BUNDLE HBM0 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_ONLY}
dict set maxi_interface_dict HBM1 { CHANNEL_NUM 0 BUNDLE HBM1 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_ONLY}
dict set maxi_interface_dict HBM2 { CHANNEL_NUM 0 BUNDLE HBM2 NUM_READ_OUTSTANDING 16 NUM_WRITE_OUTSTANDING 16 MAX_READ_BURST_LENGTH 16 MAX_WRITE_BURST_LENGTH 16 READ_WRITE_MODE READ_ONLY}

# RTL port scheduling information:
set fifoSchedulingInfoList { 
}

# RTL bus port read request latency information:
set busReadReqLatencyList { 
	{ HBM0 64 }
	{ HBM1 1 }
	{ HBM2 1 }
}

# RTL bus port write response latency information:
set busWriteResLatencyList { 
	{ HBM0 64 }
	{ HBM1 1 }
	{ HBM2 1 }
}

# RTL array port load latency information:
set memoryLoadLatencyList { 
}
