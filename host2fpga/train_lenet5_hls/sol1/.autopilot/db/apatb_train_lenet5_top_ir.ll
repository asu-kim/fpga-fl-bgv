; ModuleID = '/home/megan/Vitis-AI/accelerator/fpga-fl-bgv/host2fpga/train_lenet5_hls/sol1/.autopilot/db/a.g.ld.5.gdce.bc'
source_filename = "llvm-link"
target datalayout = "e-m:e-i64:64-i128:128-i256:256-i512:512-i1024:1024-i2048:2048-i4096:4096-n8:16:32:64-S128-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "fpga64-xilinx-none"

; Function Attrs: noinline
define void @apatb_train_lenet5_top_ir(float* noalias nocapture nonnull readonly "maxi" %image, float* noalias nocapture nonnull readonly "maxi" %conv1_weight, float* noalias nocapture nonnull readonly "maxi" %conv1_bias, float* noalias nocapture nonnull readonly "maxi" %conv2_in, float* noalias nocapture nonnull readonly "maxi" %conv2_weight, float* noalias nocapture nonnull readonly "maxi" %conv2_bias, float* noalias nocapture nonnull readonly "maxi" %fc1_in, float* noalias nocapture nonnull readonly "maxi" %fc1_weight, float* noalias nocapture nonnull readonly "maxi" %fc1_bias, float* noalias nocapture nonnull readonly "maxi" %fc2_in, float* noalias nocapture nonnull readonly "maxi" %fc2_weight, float* noalias nocapture nonnull readonly "maxi" %fc2_bias, float* noalias nocapture nonnull readonly "maxi" %fc3_in, float* noalias nocapture nonnull readonly "maxi" %fc3_weight, float* noalias nocapture nonnull readonly "maxi" %fc3_bias, float* noalias nocapture nonnull readonly "maxi" %probs, float* noalias nocapture nonnull readonly "maxi" %label) local_unnamed_addr #0 {
entry:
  %image_copy = alloca [784 x float], align 512
  %conv1_weight_copy = alloca [150 x float], align 512
  %conv1_bias_copy = alloca [6 x float], align 512
  %malloccall = tail call i8* @malloc(i64 13824)
  %conv2_in_copy = bitcast i8* %malloccall to [3456 x float]*
  %malloccall1 = tail call i8* @malloc(i64 9600)
  %conv2_weight_copy = bitcast i8* %malloccall1 to [2400 x float]*
  %conv2_bias_copy = alloca [16 x float], align 512
  %fc1_in_copy = alloca [256 x float], align 512
  %malloccall2 = tail call i8* @malloc(i64 122880)
  %fc1_weight_copy = bitcast i8* %malloccall2 to [30720 x float]*
  %fc1_bias_copy = alloca [120 x float], align 512
  %fc2_in_copy = alloca [120 x float], align 512
  %malloccall3 = tail call i8* @malloc(i64 40320)
  %fc2_weight_copy = bitcast i8* %malloccall3 to [10080 x float]*
  %fc2_bias_copy = alloca [84 x float], align 512
  %fc3_in_copy = alloca [84 x float], align 512
  %fc3_weight_copy = alloca [840 x float], align 512
  %fc3_bias_copy = alloca [10 x float], align 512
  %probs_copy = alloca float, align 512
  %label_copy = alloca float, align 512
  %0 = bitcast float* %image to [784 x float]*
  %1 = bitcast float* %conv1_weight to [150 x float]*
  %2 = bitcast float* %conv1_bias to [6 x float]*
  %3 = bitcast float* %conv2_in to [3456 x float]*
  %4 = bitcast float* %conv2_weight to [2400 x float]*
  %5 = bitcast float* %conv2_bias to [16 x float]*
  %6 = bitcast float* %fc1_in to [256 x float]*
  %7 = bitcast float* %fc1_weight to [30720 x float]*
  %8 = bitcast float* %fc1_bias to [120 x float]*
  %9 = bitcast float* %fc2_in to [120 x float]*
  %10 = bitcast float* %fc2_weight to [10080 x float]*
  %11 = bitcast float* %fc2_bias to [84 x float]*
  %12 = bitcast float* %fc3_in to [84 x float]*
  %13 = bitcast float* %fc3_weight to [840 x float]*
  %14 = bitcast float* %fc3_bias to [10 x float]*
  call fastcc void @copy_in([784 x float]* nonnull %0, [784 x float]* nonnull align 512 %image_copy, [150 x float]* nonnull %1, [150 x float]* nonnull align 512 %conv1_weight_copy, [6 x float]* nonnull %2, [6 x float]* nonnull align 512 %conv1_bias_copy, [3456 x float]* nonnull %3, [3456 x float]* %conv2_in_copy, [2400 x float]* nonnull %4, [2400 x float]* %conv2_weight_copy, [16 x float]* nonnull %5, [16 x float]* nonnull align 512 %conv2_bias_copy, [256 x float]* nonnull %6, [256 x float]* nonnull align 512 %fc1_in_copy, [30720 x float]* nonnull %7, [30720 x float]* %fc1_weight_copy, [120 x float]* nonnull %8, [120 x float]* nonnull align 512 %fc1_bias_copy, [120 x float]* nonnull %9, [120 x float]* nonnull align 512 %fc2_in_copy, [10080 x float]* nonnull %10, [10080 x float]* %fc2_weight_copy, [84 x float]* nonnull %11, [84 x float]* nonnull align 512 %fc2_bias_copy, [84 x float]* nonnull %12, [84 x float]* nonnull align 512 %fc3_in_copy, [840 x float]* nonnull %13, [840 x float]* nonnull align 512 %fc3_weight_copy, [10 x float]* nonnull %14, [10 x float]* nonnull align 512 %fc3_bias_copy, float* nonnull %probs, float* nonnull align 512 %probs_copy, float* nonnull %label, float* nonnull align 512 %label_copy)
  call void @apatb_train_lenet5_top_hw([784 x float]* %image_copy, [150 x float]* %conv1_weight_copy, [6 x float]* %conv1_bias_copy, [3456 x float]* %conv2_in_copy, [2400 x float]* %conv2_weight_copy, [16 x float]* %conv2_bias_copy, [256 x float]* %fc1_in_copy, [30720 x float]* %fc1_weight_copy, [120 x float]* %fc1_bias_copy, [120 x float]* %fc2_in_copy, [10080 x float]* %fc2_weight_copy, [84 x float]* %fc2_bias_copy, [84 x float]* %fc3_in_copy, [840 x float]* %fc3_weight_copy, [10 x float]* %fc3_bias_copy, float* %probs_copy, float* %label_copy)
  call void @copy_back([784 x float]* %0, [784 x float]* %image_copy, [150 x float]* %1, [150 x float]* %conv1_weight_copy, [6 x float]* %2, [6 x float]* %conv1_bias_copy, [3456 x float]* %3, [3456 x float]* %conv2_in_copy, [2400 x float]* %4, [2400 x float]* %conv2_weight_copy, [16 x float]* %5, [16 x float]* %conv2_bias_copy, [256 x float]* %6, [256 x float]* %fc1_in_copy, [30720 x float]* %7, [30720 x float]* %fc1_weight_copy, [120 x float]* %8, [120 x float]* %fc1_bias_copy, [120 x float]* %9, [120 x float]* %fc2_in_copy, [10080 x float]* %10, [10080 x float]* %fc2_weight_copy, [84 x float]* %11, [84 x float]* %fc2_bias_copy, [84 x float]* %12, [84 x float]* %fc3_in_copy, [840 x float]* %13, [840 x float]* %fc3_weight_copy, [10 x float]* %14, [10 x float]* %fc3_bias_copy, float* %probs, float* %probs_copy, float* %label, float* %label_copy)
  tail call void @free(i8* %malloccall)
  tail call void @free(i8* %malloccall1)
  tail call void @free(i8* %malloccall2)
  tail call void @free(i8* %malloccall3)
  ret void
}

declare noalias i8* @malloc(i64) local_unnamed_addr

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_in([784 x float]* noalias readonly, [784 x float]* noalias align 512, [150 x float]* noalias readonly, [150 x float]* noalias align 512, [6 x float]* noalias readonly, [6 x float]* noalias align 512, [3456 x float]* noalias readonly, [3456 x float]* noalias, [2400 x float]* noalias readonly, [2400 x float]* noalias, [16 x float]* noalias readonly, [16 x float]* noalias align 512, [256 x float]* noalias readonly, [256 x float]* noalias align 512, [30720 x float]* noalias readonly, [30720 x float]* noalias, [120 x float]* noalias readonly, [120 x float]* noalias align 512, [120 x float]* noalias readonly, [120 x float]* noalias align 512, [10080 x float]* noalias readonly, [10080 x float]* noalias, [84 x float]* noalias readonly, [84 x float]* noalias align 512, [84 x float]* noalias readonly, [84 x float]* noalias align 512, [840 x float]* noalias readonly, [840 x float]* noalias align 512, [10 x float]* noalias readonly, [10 x float]* noalias align 512, float* noalias readonly, float* noalias align 512, float* noalias readonly, float* noalias align 512) unnamed_addr #1 {
entry:
  call fastcc void @onebyonecpy_hls.p0a784f32([784 x float]* align 512 %1, [784 x float]* %0)
  call fastcc void @onebyonecpy_hls.p0a150f32([150 x float]* align 512 %3, [150 x float]* %2)
  call fastcc void @onebyonecpy_hls.p0a6f32([6 x float]* align 512 %5, [6 x float]* %4)
  call fastcc void @onebyonecpy_hls.p0a3456f32([3456 x float]* %7, [3456 x float]* %6)
  call fastcc void @onebyonecpy_hls.p0a2400f32([2400 x float]* %9, [2400 x float]* %8)
  call fastcc void @onebyonecpy_hls.p0a16f32([16 x float]* align 512 %11, [16 x float]* %10)
  call fastcc void @onebyonecpy_hls.p0a256f32([256 x float]* align 512 %13, [256 x float]* %12)
  call fastcc void @onebyonecpy_hls.p0a30720f32([30720 x float]* %15, [30720 x float]* %14)
  call fastcc void @onebyonecpy_hls.p0a120f32([120 x float]* align 512 %17, [120 x float]* %16)
  call fastcc void @onebyonecpy_hls.p0a120f32([120 x float]* align 512 %19, [120 x float]* %18)
  call fastcc void @onebyonecpy_hls.p0a10080f32([10080 x float]* %21, [10080 x float]* %20)
  call fastcc void @onebyonecpy_hls.p0a84f32([84 x float]* align 512 %23, [84 x float]* %22)
  call fastcc void @onebyonecpy_hls.p0a84f32([84 x float]* align 512 %25, [84 x float]* %24)
  call fastcc void @onebyonecpy_hls.p0a840f32([840 x float]* align 512 %27, [840 x float]* %26)
  call fastcc void @onebyonecpy_hls.p0a10f32([10 x float]* align 512 %29, [10 x float]* %28)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %31, float* %30)
  call fastcc void @onebyonecpy_hls.p0f32(float* align 512 %33, float* %32)
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a784f32([784 x float]* noalias align 512 %dst, [784 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [784 x float]* %dst, null
  %1 = icmp eq [784 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a784f32([784 x float]* nonnull %dst, [784 x float]* nonnull %src, i64 784)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a784f32([784 x float]* %dst, [784 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [784 x float]* %src, null
  %1 = icmp eq [784 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [784 x float], [784 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [784 x float], [784 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a150f32([150 x float]* noalias align 512 %dst, [150 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [150 x float]* %dst, null
  %1 = icmp eq [150 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a150f32([150 x float]* nonnull %dst, [150 x float]* nonnull %src, i64 150)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a150f32([150 x float]* %dst, [150 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [150 x float]* %src, null
  %1 = icmp eq [150 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [150 x float], [150 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [150 x float], [150 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a6f32([6 x float]* noalias align 512 %dst, [6 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [6 x float]* %dst, null
  %1 = icmp eq [6 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a6f32([6 x float]* nonnull %dst, [6 x float]* nonnull %src, i64 6)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a6f32([6 x float]* %dst, [6 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [6 x float]* %src, null
  %1 = icmp eq [6 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [6 x float], [6 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [6 x float], [6 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a3456f32([3456 x float]* noalias %dst, [3456 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [3456 x float]* %dst, null
  %1 = icmp eq [3456 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a3456f32([3456 x float]* nonnull %dst, [3456 x float]* nonnull %src, i64 3456)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a3456f32([3456 x float]* %dst, [3456 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [3456 x float]* %src, null
  %1 = icmp eq [3456 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [3456 x float], [3456 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [3456 x float], [3456 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a2400f32([2400 x float]* noalias %dst, [2400 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [2400 x float]* %dst, null
  %1 = icmp eq [2400 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a2400f32([2400 x float]* nonnull %dst, [2400 x float]* nonnull %src, i64 2400)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a2400f32([2400 x float]* %dst, [2400 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [2400 x float]* %src, null
  %1 = icmp eq [2400 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [2400 x float], [2400 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [2400 x float], [2400 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a16f32([16 x float]* noalias align 512 %dst, [16 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [16 x float]* %dst, null
  %1 = icmp eq [16 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a16f32([16 x float]* nonnull %dst, [16 x float]* nonnull %src, i64 16)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a16f32([16 x float]* %dst, [16 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [16 x float]* %src, null
  %1 = icmp eq [16 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [16 x float], [16 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [16 x float], [16 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a256f32([256 x float]* noalias align 512 %dst, [256 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [256 x float]* %dst, null
  %1 = icmp eq [256 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a256f32([256 x float]* nonnull %dst, [256 x float]* nonnull %src, i64 256)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a256f32([256 x float]* %dst, [256 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [256 x float]* %src, null
  %1 = icmp eq [256 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [256 x float], [256 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [256 x float], [256 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a30720f32([30720 x float]* noalias %dst, [30720 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [30720 x float]* %dst, null
  %1 = icmp eq [30720 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a30720f32([30720 x float]* nonnull %dst, [30720 x float]* nonnull %src, i64 30720)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a30720f32([30720 x float]* %dst, [30720 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [30720 x float]* %src, null
  %1 = icmp eq [30720 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [30720 x float], [30720 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [30720 x float], [30720 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a120f32([120 x float]* noalias align 512 %dst, [120 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [120 x float]* %dst, null
  %1 = icmp eq [120 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a120f32([120 x float]* nonnull %dst, [120 x float]* nonnull %src, i64 120)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a120f32([120 x float]* %dst, [120 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [120 x float]* %src, null
  %1 = icmp eq [120 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [120 x float], [120 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [120 x float], [120 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a10080f32([10080 x float]* noalias %dst, [10080 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [10080 x float]* %dst, null
  %1 = icmp eq [10080 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a10080f32([10080 x float]* nonnull %dst, [10080 x float]* nonnull %src, i64 10080)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a10080f32([10080 x float]* %dst, [10080 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [10080 x float]* %src, null
  %1 = icmp eq [10080 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [10080 x float], [10080 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [10080 x float], [10080 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a84f32([84 x float]* noalias align 512 %dst, [84 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [84 x float]* %dst, null
  %1 = icmp eq [84 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a84f32([84 x float]* nonnull %dst, [84 x float]* nonnull %src, i64 84)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a84f32([84 x float]* %dst, [84 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [84 x float]* %src, null
  %1 = icmp eq [84 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [84 x float], [84 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [84 x float], [84 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a840f32([840 x float]* noalias align 512 %dst, [840 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [840 x float]* %dst, null
  %1 = icmp eq [840 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a840f32([840 x float]* nonnull %dst, [840 x float]* nonnull %src, i64 840)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a840f32([840 x float]* %dst, [840 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [840 x float]* %src, null
  %1 = icmp eq [840 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [840 x float], [840 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [840 x float], [840 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0a10f32([10 x float]* noalias align 512 %dst, [10 x float]* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq [10 x float]* %dst, null
  %1 = icmp eq [10 x float]* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  call void @arraycpy_hls.p0a10f32([10 x float]* nonnull %dst, [10 x float]* nonnull %src, i64 10)
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define void @arraycpy_hls.p0a10f32([10 x float]* %dst, [10 x float]* readonly %src, i64 %num) local_unnamed_addr #3 {
entry:
  %0 = icmp eq [10 x float]* %src, null
  %1 = icmp eq [10 x float]* %dst, null
  %2 = or i1 %1, %0
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %for.loop.cond1 = icmp sgt i64 %num, 0
  br i1 %for.loop.cond1, label %for.loop.lr.ph, label %copy.split

for.loop.lr.ph:                                   ; preds = %copy
  br label %for.loop

for.loop:                                         ; preds = %for.loop, %for.loop.lr.ph
  %for.loop.idx2 = phi i64 [ 0, %for.loop.lr.ph ], [ %for.loop.idx.next, %for.loop ]
  %dst.addr = getelementptr [10 x float], [10 x float]* %dst, i64 0, i64 %for.loop.idx2
  %src.addr = getelementptr [10 x float], [10 x float]* %src, i64 0, i64 %for.loop.idx2
  %3 = load float, float* %src.addr, align 4
  store float %3, float* %dst.addr, align 4
  %for.loop.idx.next = add nuw nsw i64 %for.loop.idx2, 1
  %exitcond = icmp ne i64 %for.loop.idx.next, %num
  br i1 %exitcond, label %for.loop, label %copy.split

copy.split:                                       ; preds = %for.loop, %copy
  br label %ret

ret:                                              ; preds = %copy.split, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @onebyonecpy_hls.p0f32(float* noalias align 512 %dst, float* noalias readonly %src) unnamed_addr #2 {
entry:
  %0 = icmp eq float* %dst, null
  %1 = icmp eq float* %src, null
  %2 = or i1 %0, %1
  br i1 %2, label %ret, label %copy

copy:                                             ; preds = %entry
  %3 = load float, float* %src, align 4
  store float %3, float* %dst, align 512
  br label %ret

ret:                                              ; preds = %copy, %entry
  ret void
}

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_out([784 x float]* noalias, [784 x float]* noalias readonly align 512, [150 x float]* noalias, [150 x float]* noalias readonly align 512, [6 x float]* noalias, [6 x float]* noalias readonly align 512, [3456 x float]* noalias, [3456 x float]* noalias readonly, [2400 x float]* noalias, [2400 x float]* noalias readonly, [16 x float]* noalias, [16 x float]* noalias readonly align 512, [256 x float]* noalias, [256 x float]* noalias readonly align 512, [30720 x float]* noalias, [30720 x float]* noalias readonly, [120 x float]* noalias, [120 x float]* noalias readonly align 512, [120 x float]* noalias, [120 x float]* noalias readonly align 512, [10080 x float]* noalias, [10080 x float]* noalias readonly, [84 x float]* noalias, [84 x float]* noalias readonly align 512, [84 x float]* noalias, [84 x float]* noalias readonly align 512, [840 x float]* noalias, [840 x float]* noalias readonly align 512, [10 x float]* noalias, [10 x float]* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512) unnamed_addr #4 {
entry:
  call fastcc void @onebyonecpy_hls.p0a784f32([784 x float]* %0, [784 x float]* align 512 %1)
  call fastcc void @onebyonecpy_hls.p0a150f32([150 x float]* %2, [150 x float]* align 512 %3)
  call fastcc void @onebyonecpy_hls.p0a6f32([6 x float]* %4, [6 x float]* align 512 %5)
  call fastcc void @onebyonecpy_hls.p0a3456f32([3456 x float]* %6, [3456 x float]* %7)
  call fastcc void @onebyonecpy_hls.p0a2400f32([2400 x float]* %8, [2400 x float]* %9)
  call fastcc void @onebyonecpy_hls.p0a16f32([16 x float]* %10, [16 x float]* align 512 %11)
  call fastcc void @onebyonecpy_hls.p0a256f32([256 x float]* %12, [256 x float]* align 512 %13)
  call fastcc void @onebyonecpy_hls.p0a30720f32([30720 x float]* %14, [30720 x float]* %15)
  call fastcc void @onebyonecpy_hls.p0a120f32([120 x float]* %16, [120 x float]* align 512 %17)
  call fastcc void @onebyonecpy_hls.p0a120f32([120 x float]* %18, [120 x float]* align 512 %19)
  call fastcc void @onebyonecpy_hls.p0a10080f32([10080 x float]* %20, [10080 x float]* %21)
  call fastcc void @onebyonecpy_hls.p0a84f32([84 x float]* %22, [84 x float]* align 512 %23)
  call fastcc void @onebyonecpy_hls.p0a84f32([84 x float]* %24, [84 x float]* align 512 %25)
  call fastcc void @onebyonecpy_hls.p0a840f32([840 x float]* %26, [840 x float]* align 512 %27)
  call fastcc void @onebyonecpy_hls.p0a10f32([10 x float]* %28, [10 x float]* align 512 %29)
  call fastcc void @onebyonecpy_hls.p0f32(float* %30, float* align 512 %31)
  call fastcc void @onebyonecpy_hls.p0f32(float* %32, float* align 512 %33)
  ret void
}

declare void @free(i8*) local_unnamed_addr

declare void @apatb_train_lenet5_top_hw([784 x float]*, [150 x float]*, [6 x float]*, [3456 x float]*, [2400 x float]*, [16 x float]*, [256 x float]*, [30720 x float]*, [120 x float]*, [120 x float]*, [10080 x float]*, [84 x float]*, [84 x float]*, [840 x float]*, [10 x float]*, float*, float*)

; Function Attrs: argmemonly noinline norecurse willreturn
define internal fastcc void @copy_back([784 x float]* noalias, [784 x float]* noalias readonly align 512, [150 x float]* noalias, [150 x float]* noalias readonly align 512, [6 x float]* noalias, [6 x float]* noalias readonly align 512, [3456 x float]* noalias, [3456 x float]* noalias readonly, [2400 x float]* noalias, [2400 x float]* noalias readonly, [16 x float]* noalias, [16 x float]* noalias readonly align 512, [256 x float]* noalias, [256 x float]* noalias readonly align 512, [30720 x float]* noalias, [30720 x float]* noalias readonly, [120 x float]* noalias, [120 x float]* noalias readonly align 512, [120 x float]* noalias, [120 x float]* noalias readonly align 512, [10080 x float]* noalias, [10080 x float]* noalias readonly, [84 x float]* noalias, [84 x float]* noalias readonly align 512, [84 x float]* noalias, [84 x float]* noalias readonly align 512, [840 x float]* noalias, [840 x float]* noalias readonly align 512, [10 x float]* noalias, [10 x float]* noalias readonly align 512, float* noalias, float* noalias readonly align 512, float* noalias, float* noalias readonly align 512) unnamed_addr #4 {
entry:
  ret void
}

define void @train_lenet5_top_hw_stub_wrapper([784 x float]*, [150 x float]*, [6 x float]*, [3456 x float]*, [2400 x float]*, [16 x float]*, [256 x float]*, [30720 x float]*, [120 x float]*, [120 x float]*, [10080 x float]*, [84 x float]*, [84 x float]*, [840 x float]*, [10 x float]*, float*, float*) #5 {
entry:
  call void @copy_out([784 x float]* null, [784 x float]* %0, [150 x float]* null, [150 x float]* %1, [6 x float]* null, [6 x float]* %2, [3456 x float]* null, [3456 x float]* %3, [2400 x float]* null, [2400 x float]* %4, [16 x float]* null, [16 x float]* %5, [256 x float]* null, [256 x float]* %6, [30720 x float]* null, [30720 x float]* %7, [120 x float]* null, [120 x float]* %8, [120 x float]* null, [120 x float]* %9, [10080 x float]* null, [10080 x float]* %10, [84 x float]* null, [84 x float]* %11, [84 x float]* null, [84 x float]* %12, [840 x float]* null, [840 x float]* %13, [10 x float]* null, [10 x float]* %14, float* null, float* %15, float* null, float* %16)
  %17 = bitcast [784 x float]* %0 to float*
  %18 = bitcast [150 x float]* %1 to float*
  %19 = bitcast [6 x float]* %2 to float*
  %20 = bitcast [3456 x float]* %3 to float*
  %21 = bitcast [2400 x float]* %4 to float*
  %22 = bitcast [16 x float]* %5 to float*
  %23 = bitcast [256 x float]* %6 to float*
  %24 = bitcast [30720 x float]* %7 to float*
  %25 = bitcast [120 x float]* %8 to float*
  %26 = bitcast [120 x float]* %9 to float*
  %27 = bitcast [10080 x float]* %10 to float*
  %28 = bitcast [84 x float]* %11 to float*
  %29 = bitcast [84 x float]* %12 to float*
  %30 = bitcast [840 x float]* %13 to float*
  %31 = bitcast [10 x float]* %14 to float*
  call void @train_lenet5_top_hw_stub(float* %17, float* %18, float* %19, float* %20, float* %21, float* %22, float* %23, float* %24, float* %25, float* %26, float* %27, float* %28, float* %29, float* %30, float* %31, float* %15, float* %16)
  call void @copy_in([784 x float]* null, [784 x float]* %0, [150 x float]* null, [150 x float]* %1, [6 x float]* null, [6 x float]* %2, [3456 x float]* null, [3456 x float]* %3, [2400 x float]* null, [2400 x float]* %4, [16 x float]* null, [16 x float]* %5, [256 x float]* null, [256 x float]* %6, [30720 x float]* null, [30720 x float]* %7, [120 x float]* null, [120 x float]* %8, [120 x float]* null, [120 x float]* %9, [10080 x float]* null, [10080 x float]* %10, [84 x float]* null, [84 x float]* %11, [84 x float]* null, [84 x float]* %12, [840 x float]* null, [840 x float]* %13, [10 x float]* null, [10 x float]* %14, float* null, float* %15, float* null, float* %16)
  ret void
}

declare void @train_lenet5_top_hw_stub(float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly, float* noalias nocapture nonnull readonly)

attributes #0 = { noinline "fpga.wrapper.func"="wrapper" }
attributes #1 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyin" }
attributes #2 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="onebyonecpy_hls" }
attributes #3 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="arraycpy_hls" }
attributes #4 = { argmemonly noinline norecurse willreturn "fpga.wrapper.func"="copyout" }
attributes #5 = { "fpga.wrapper.func"="stub" }

!llvm.dbg.cu = !{}
!llvm.ident = !{!0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0, !0}
!llvm.module.flags = !{!1, !2, !3}
!blackbox_cfg = !{!4}

!0 = !{!"clang version 7.0.0 "}
!1 = !{i32 2, !"Dwarf Version", i32 4}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{}
