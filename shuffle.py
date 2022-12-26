import random

#eg: shuffle_split(80, "xyz.csv")  80= 80% train, 20% Test. xyz.csv= source_file

def shuffle_split(train_size, filename):
	data=open(filename, "r")

	train1=open("/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/train.csv", "w")
	test1= open("/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/test.csv", "w")
	read_data= data.readlines()
	random.shuffle(read_data)
	length = (len(read_data)*train_size)/100
	b=[]
	c=[]
	for x,y in enumerate(read_data):
		if x < length:
			b.append(y)
		else:
			c.append(y)

	train1.writelines(b)
	test1.writelines(c)

	train1.close()
	test1.close()
	data.close()

shuffle_split(80, "/home/dung/ivsr-s2p/kitti_airsim2412/2022-12-23-18-30-33/derived_label.csv")