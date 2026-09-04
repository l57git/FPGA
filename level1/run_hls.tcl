set project_root [file normalize [file dirname [info script]]]
set hls_workspace [file join $project_root hls_work]
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

file mkdir $hls_workspace
cd $hls_workspace

open_project -reset lenet_level1_hls
set_top lenet_accel
add_files [file join $project_root src lenet.cpp]
add_files [file join $project_root src lenet.hpp]
add_files -tb [file join $project_root tb tb_lenet.cpp]

open_solution -reset solution1
set_part {xc7z020clg400-1}
create_clock -period 10 -name default

if {$skip_csim} {
    puts "LENET_SKIP_CSIM=1; skipping csim_design."
} elseif {[file exists $accuracy_blob]} {
    puts "LENET_ACCURACY_BLOB=$accuracy_blob"
    set csim_argv $accuracy_blob
    if {$result_csv ne ""} {
        append csim_argv " " $result_csv
        puts "LENET_RESULT_CSV=$result_csv"
    }
    if {$accuracy_threshold ne ""} {
        append csim_argv " " $accuracy_threshold
        puts "LENET_ACCURACY_THRESHOLD=$accuracy_threshold"
    }
    csim_design -argv $csim_argv
} else {
    puts "data/lenet_accuracy.bin not found; running smoke test only."
    csim_design
}

if {$skip_synth} {
    puts "LENET_SKIP_SYNTH=1; skipping csynth_design."
} else {
    csynth_design
}
exit
