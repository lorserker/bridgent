import sys


def main():
	n = int(sys.argv[1])
	radix = int(sys.argv[2])

	fins = [open('script_%d.sh' % i, 'w') for i in range(radix)]
	for fin in fins:
		fin.write('export PYTHONIOENCODING=ascii\n\n')

	for i in range(n):
		r = i % radix
		fins[r].write('python gen_double_dummy.py 10000 | gzip > data/out_%04d.gz' % i)
		fins[r].write('\n')

	for fin in fins:
		fin.close()


if __name__ == '__main__':
	main()