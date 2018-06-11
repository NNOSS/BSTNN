#Author: Nick Steelman
#Date: 6/11/18
#gns126@gmail.com
#cleanestmink.com

def set_parameters(block_info, block_info_parent):
    set_learning_rate(block_info, block_info_parent)
    set_beta1(block_info, block_info_parent)
    set_convolutions(block_info, block_info_parent)
    set_fully_connected_size(block_info, block_info_parent)

def set_learning_rate(block_info, block_info_parent):
    block_info.learning_rate = block_info_parent.learning_rate

def set_beta1(block_info, block_info_parent):
    block_info.beta1 = block_info_parent.beta1

def set_convolutions(block_info, block_info_parent):
    block_info.convolutions = block_info_parent.convolutions

def set_fully_connected_size(block_info, block_info_parent):
    block_info.fully_connected_size = block_info_parent.fully_connected_size
