from math import log10, sqrt 
import cv2 
import numpy as np 
import argparse


def PSNR(original, compressed): 
	""" Obtain the PSNR value given original and compressed images"""

	mse = np.mean((original - compressed) ** 2) 
	if(mse == 0): # MSE is zero means no noise is present in the signal . 
				# Therefore PSNR have no importance. 
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse)) 
	return psnr 

def main(args): 

	original = cv2.imread(args.output)

	compressed = cv2.imread(args.input)

	value = PSNR(original, compressed) 
	print(f"PSNR value is {value} dB") 
	
if __name__ == "__main__": 
	parser = argparse.ArgumentParser()
	parser.add_argument("input")
	parser.add_argument("output")
	args = parser.parse_args()
	main(args) 
