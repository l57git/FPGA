set project_root [file normalize [file dirname [info script]]]
set hls_workspace [file join $project_root hls_work]
if {[info exists ::env(LENET_HLS_WORKSPACE)] && $::env(LENET_HLS_WORKSPACE) ne ""} {
    set hls_workspace [file normalize $::env(LENET_HLS_WORKSPACE)]
}
set data_width 16
if {[info exists ::env(LENET_DATA_W)] && $::env(LENET_DATA_W) ne ""} {
    set data_width $::env(LENET_DATA_W)
}
if {![string is integer -strict $data_width] || $data_width < 8 || $data_width > 16} {
    error "LENET_DATA_W must be an integer in the range 8..16; got '$data_width'"
}
set lenet_cflags "-DLENET_DATA_W=$data_width"
if {[info exists ::env(LENET_ACCURACY_BLOB)]} {
    set accuracy_blob [file normalize $::env(LENET_ACCURACY_BLOB)]
} else {
    set accuracy_blob [file join $project_root data lenet_accuracy.bin]
}
set result_csv ""
if {[info exists ::env(LENET_RESULT_CSV)]} {
    set result_csv [file normalize $::env(LENET_RESULT_CSV)]
}
set accuracy_threshold ""
if {[info exists ::env(LENET_ACCURACY_THRESHOLD)]} {
    set accuracy_threshold $::env(LENET_ACCURACY_THRESHOLD)
}
set skip_synth 0
if {[info exists ::env(LENET_SKIP_SYNTH)] && $::env(LENET_SKIP_SYNTH) eq "1"} {
    set skip_synth 1
}
set skip_csim 0
if {[info exists ::env(LENET_SKIP_CSIM)] && $::env(LENET_SKIP_CSIM) eq "1"} {
    set skip_csim 1
}
set csim_optimized 0
if {[info exists ::env(LENET_CSIM_OPT)] && $::env(LENET_CSIM_OPT) eq "1"} {
    set csim_optimized 1
}
set csim_mflags ""
if {[info exists ::env(LENET_CSIM_MFLAGS)]} {
    set csim_mflags $::env(LENET_CSIM_MFLAGS)
}

file mkdir $hls_workspace
cd $hls_workspace

puts "LENET_DATA_W=$data_width"
puts "LENET_DATA_TYPE=ap_fixed<${data_width},6,AP_RND,AP_SAT>"
puts "LENET_ACC_TYPE=ap_fixed<32,14,AP_RND,AP_SAT>"
puts "LENET_PART=xc7z020clg400-1"
puts "LENET_CLOCK_NS=10"
puts "LENET_HLS_WORKSPACE=$hls_workspace"
puts "LENET_CSIM_OPT=$csim_optimized"
puts "LENET_CSIM_MFLAGS=$csim_mflags"

open_project -reset lenet_level1_hls
set_top lenet_accel
add_files [file join $project_root src lenet.cpp] -cflags $lenet_cflags
add_files [file join $project_root src lenet.hpp] -cflags $lenet_cflags
add_files -tb [file join $project_root tb tb_lenet.cpp] -cflags $lenet_cflags

open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

if {$skip_csim} {
    puts "LENET_SKIP_CSIM=1; skipping csim_design."
} elseif {[file exists $accuracy_blob]} {
    puts "LENET_ACCURACY_BLOB=$accuracy_blob"
    set csim_argv [list $accuracy_blob]
    if {$result_csv ne ""} {
        file mkdir [file dirname $result_csv]
        lappend csim_argv $result_csv
        puts "LENET_RESULT_CSV=$result_csv"
    }
    if {$accuracy_threshold ne ""} {
        lappend csim_argv $accuracy_threshold
        puts "LENET_ACCURACY_THRESHOLD=$accuracy_threshold"
    }
    if {$csim_optimized} {
        if {$csim_mflags ne ""} {
            csim_design -O -mflags $csim_mflags -argv $csim_argv
        } else {
            csim_design -O -argv $csim_argv
        }
    } else {
        csim_design -argv $csim_argv
    }
} else {
    puts "data/lenet_accuracy.bin not found; running smoke test only."
    if {$csim_optimized} {
        csim_design -O
    } else {
        csim_design
    }
}

if {$skip_synth} {
    puts "LENET_SKIP_SYNTH=1; skipping csynth_design."
} else {
    csynth_design
}
exit
