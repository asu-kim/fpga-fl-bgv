<AutoPilot:project xmlns:AutoPilot="com.autoesl.autopilot.project" projectType="C/C++" name="train_lenet5_hls" ideType="classic" top="train_lenet5_top">
    <Simulation argv="">
        <SimFlow name="csim" setup="false" optimizeCompile="false" clean="false" ldflags="" mflags=""/>
    </Simulation>
    <files>
        <file name="../../test/test_train.cpp" sc="0" tb="1" cflags="-I../../kernels -Wno-unknown-pragmas" csimflags="" blackbox="false"/>
        <file name="kernels/train.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/train.cpp" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/mse_loss.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/update.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/fc_bwd.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/fc.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/flatten_bwd.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/flatten.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/avg_pool_bwd.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/avg_pool.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/conv2d_bwd.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/conv2d.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernel/reader.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/utils.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="kernels/aes_utils.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
        <file name="data/weights_bias_raw_params.h" sc="0" tb="false" cflags="" csimflags="" blackbox="false"/>
    </files>
    <solutions>
        <solution name="sol1" status=""/>
    </solutions>
</AutoPilot:project>

