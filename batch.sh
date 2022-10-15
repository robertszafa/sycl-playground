
# # make fpga KERNEL=static

# # make fpga KERNEL=dynamic Q_SIZE=8
# # make fpga KERNEL=dynamic Q_SIZE=4
# make fpga KERNEL=dynamic_no_forward Q_SIZE=32
# make fpga KERNEL=dynamic_no_forward Q_SIZE=64
# # make fpga KERNEL=dynamic Q_SIZE=16

# make fpga KERNEL=dynamic_no_forward Q_SIZE=4
# make fpga KERNEL=dynamic_no_forward Q_SIZE=8

# make fpga KERNEL=dynamic_no_forward Q_SIZE=16

# # make fpga KERNEL=dynamic Q_SIZE=1
# make fpga KERNEL=dynamic_no_forward Q_SIZE=1

# make fpga KERNEL=dynamic Q_SIZE=2
# make fpga KERNEL=dynamic_no_forward Q_SIZE=2
# make fpga KERNEL=dynamic Q_SIZE=1

python3 build_all.py hw > make_all.out 2>&1

echo "DONE"
