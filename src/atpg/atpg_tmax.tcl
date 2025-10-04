set design "dut"
set_atpg -timeout 60
read_netlist ./${design}.v
read_netlist ./saed90nm.v -library
run_build_model c17
run_drc
add_faults -all
run_atpg -auto_compression
rm ${design}.PATTERN
write_faults ${design}.PATTERN -all
exit