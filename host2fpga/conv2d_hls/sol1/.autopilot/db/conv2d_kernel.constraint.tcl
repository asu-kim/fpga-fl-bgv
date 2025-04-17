set clock_constraint { \
    name clk \
    module conv2d_kernel \
    port ap_clk \
    period 3 \
    uncertainty 0.81 \
}

set all_path {}

set false_path {}

