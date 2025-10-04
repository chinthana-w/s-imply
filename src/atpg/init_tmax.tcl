# init_testmax.tcl

# Hardcoded BENCH file name
set bench_file "dut.bench"

# Read the circuit from the hardcoded BENCH file
if {[catch {read_bench $bench_file} result]} {
    error "Error: Could not read file '$bench_file'. Check the file path and permissions.\nDetails: $result"
}

puts "Successfully read design from $bench_file"

# Save the session for later use by the second script
write_session -file "atpg_session.ses"