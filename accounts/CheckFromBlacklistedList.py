from numpy import genfromtxt
import numpy

def check_blacklisted(nic_no):
    # nic_no in str format like '42202-0000000-0'
    n = ''
    for i in range(len(nic_no)):
        if nic_no[i] != '-':
            n+= nic_no[i]
    n=int(n)
    nic = numpy.array(n)
    nic = nic.astype(str)


    # read csv

    blck_list = genfromtxt('C:\\Users\\HP\\Downloads\\Digitalcustomer-master-Updated\\Digitalcustomer-master-Updated\\accounts\\blacklisted_ppl_list.csv')
    blck_list = blck_list.astype(numpy.int64)
    blck_list = blck_list.astype(str)
    if nic in blck_list:
        return 0    # blacklisted
    else:
        return 1


print(check_blacklisted('36302-6932036-9'))