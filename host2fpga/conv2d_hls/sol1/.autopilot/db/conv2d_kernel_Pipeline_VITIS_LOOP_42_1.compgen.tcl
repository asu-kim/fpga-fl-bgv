# This script segment is generated automatically by AutoPilot

# clear list
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_begin
    cg_default_interface_gen_bundle_begin
    AESL_LIB_XILADAPTER::native_axis_begin
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 1 \
    name HBM0 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_HBM0 \
    op interface \
    ports { m_axi_HBM0_AWVALID { O 1 bit } m_axi_HBM0_AWREADY { I 1 bit } m_axi_HBM0_AWADDR { O 64 vector } m_axi_HBM0_AWID { O 1 vector } m_axi_HBM0_AWLEN { O 32 vector } m_axi_HBM0_AWSIZE { O 3 vector } m_axi_HBM0_AWBURST { O 2 vector } m_axi_HBM0_AWLOCK { O 2 vector } m_axi_HBM0_AWCACHE { O 4 vector } m_axi_HBM0_AWPROT { O 3 vector } m_axi_HBM0_AWQOS { O 4 vector } m_axi_HBM0_AWREGION { O 4 vector } m_axi_HBM0_AWUSER { O 1 vector } m_axi_HBM0_WVALID { O 1 bit } m_axi_HBM0_WREADY { I 1 bit } m_axi_HBM0_WDATA { O 64 vector } m_axi_HBM0_WSTRB { O 8 vector } m_axi_HBM0_WLAST { O 1 bit } m_axi_HBM0_WID { O 1 vector } m_axi_HBM0_WUSER { O 1 vector } m_axi_HBM0_ARVALID { O 1 bit } m_axi_HBM0_ARREADY { I 1 bit } m_axi_HBM0_ARADDR { O 64 vector } m_axi_HBM0_ARID { O 1 vector } m_axi_HBM0_ARLEN { O 32 vector } m_axi_HBM0_ARSIZE { O 3 vector } m_axi_HBM0_ARBURST { O 2 vector } m_axi_HBM0_ARLOCK { O 2 vector } m_axi_HBM0_ARCACHE { O 4 vector } m_axi_HBM0_ARPROT { O 3 vector } m_axi_HBM0_ARQOS { O 4 vector } m_axi_HBM0_ARREGION { O 4 vector } m_axi_HBM0_ARUSER { O 1 vector } m_axi_HBM0_RVALID { I 1 bit } m_axi_HBM0_RREADY { O 1 bit } m_axi_HBM0_RDATA { I 64 vector } m_axi_HBM0_RLAST { I 1 bit } m_axi_HBM0_RID { I 1 vector } m_axi_HBM0_RFIFONUM { I 9 vector } m_axi_HBM0_RUSER { I 1 vector } m_axi_HBM0_RRESP { I 2 vector } m_axi_HBM0_BVALID { I 1 bit } m_axi_HBM0_BREADY { O 1 bit } m_axi_HBM0_BRESP { I 2 vector } m_axi_HBM0_BID { I 1 vector } m_axi_HBM0_BUSER { I 1 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id 2 \
    name sext_ln42 \
    type other \
    dir I \
    reset_level 1 \
    sync_rst true \
    corename dc_sext_ln42 \
    op interface \
    ports { sext_ln42 { I 61 vector } } \
} "
}

# Direct connection:
if {${::AESL::PGuard_autoexp_gen}} {
eval "cg_default_interface_gen_dc { \
    id -1 \
    name ap_ctrl \
    type ap_ctrl \
    reset_level 1 \
    sync_rst true \
    corename ap_ctrl \
    op interface \
    ports { ap_start { I 1 bit } ap_ready { O 1 bit } ap_done { O 1 bit } ap_idle { O 1 bit } } \
} "
}


# Adapter definition:
set PortName ap_clk
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_clock] == "cg_default_interface_gen_clock"} {
eval "cg_default_interface_gen_clock { \
    id -2 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_clk \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-113\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}


# Adapter definition:
set PortName ap_rst
set DataWd 1 
if {${::AESL::PGuard_autoexp_gen}} {
if {[info proc cg_default_interface_gen_reset] == "cg_default_interface_gen_reset"} {
eval "cg_default_interface_gen_reset { \
    id -3 \
    name ${PortName} \
    reset_level 1 \
    sync_rst true \
    corename apif_ap_rst \
    data_wd ${DataWd} \
    op interface \
}"
} else {
puts "@W \[IMPL-114\] Cannot find bus interface model in the library. Ignored generation of bus interface for '${PortName}'"
}
}



# merge
if {${::AESL::PGuard_autoexp_gen}} {
    cg_default_interface_gen_dc_end
    cg_default_interface_gen_bundle_end
    AESL_LIB_XILADAPTER::native_axis_end
}


