import csv

import glob
import os
import argparse



def main(args):

	output_csv = args.output_csv

	imgs = glob.glob(os.path.join(args.exp_path, '*.jpg'))
	imgs_id = [i.split('/')[-1].split('.')[0] for i in imgs]
	imgs_id.sort()

	person_imgs = {}
	for _id in imgs_id:
		if _id[:-9] not in person_imgs:
			person_imgs[_id[:-9]] = []
		person_imgs[_id[:-9]].append(_id)
	print('total {} images'.format(len(imgs_id)))
	print('total {} cases'.format(len(person_imgs)))


	labels = glob.glob(os.path.join(args.exp_path, 'labels', '*.txt'))
	labels_id = [i.split('/')[-1].split('.')[0] for i in labels]
	labels_id.sort()

	person_labebls = {}
	for _id in labels_id:
		if _id[:-9] not in person_labebls:
			person_labebls[_id[:-9]] = []
		person_labebls[_id[:-9]].append(_id)
	print('total {} labels'.format(len(labels_id)))
	print('total {} positive cases before post-process'.format(len(person_labebls)))


	### writing csv ###
	with open(output_csv, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['id', 'label', 'coords'])

		for pid in person_imgs:
			person_imgs_id = person_imgs[pid]

			print('---- pid:', pid, '------')
			if pid in person_labebls:

				### MIN GATE
				if len(person_labebls[pid]) <= 3:
					### predicts 0
					for img_id in person_imgs_id:
						writer.writerow([img_id, 0, ''])
				else:
					### predicts 1, -1
					persons = person_labebls[pid]

					len_persons = len(person_imgs_id)
					if args.window:
						windows = []
						for i in range(len_persons):
							if i == 0:
								windows.append([0, 1])
							elif i == len_persons -1:
								windows.append([len_persons-2, len_persons-1])
							else:
								windows.append([i-1, i, i+1])

						out_write = []
						out_1 = 0
						for w in windows:
							if w[0] == 0 and len(w)==2:
								id_1 = person_imgs_id[w[0]]
								id_2 = person_imgs_id[w[1]]
								label_1 = os.path.join(args.exp_path, 'labels', id_1+'.txt')
								label_2 = os.path.join(args.exp_path, 'labels', id_2+'.txt')

								if not os.path.isfile(label_2):
									out_write.append([person_imgs_id[w[0]], -1, ''])
								else:
									if os.path.isfile(label_1):
										f1 = open(label_1,'r')
										ls1 = f1.read().splitlines()
										f1.close()

										corrds = ''
										for l in ls1:
											l = list(map(float, l.split(' ')))
											x0, y0 = int(l[1]*512), int(l[2]*512)
											corrds += str(x0) + ' ' + str(y0) + ' '

										out_write.append([person_imgs_id[w[0]], 1, corrds[:-1]])
										out_1 += 1
									else:
										out_write.append([person_imgs_id[w[0]], -1, ''])

							elif w[-1] == len_persons-1 and len(w)==2:
								id_1 = person_imgs_id[w[0]]
								id_2 = person_imgs_id[w[1]]
								label_1 = os.path.join(args.exp_path, 'labels', id_1+'.txt')
								label_2 = os.path.join(args.exp_path, 'labels', id_2+'.txt')

								if not os.path.isfile(label_1):
									out_write.append([person_imgs_id[w[1]], -1, ''])
								else:
									if os.path.isfile(label_2):
										f2 = open(label_2,'r')
										ls2 = f2.read().splitlines()
										f2.close()

										corrds = ''
										for l in ls2:
											l = list(map(float, l.split(' ')))
											x0, y0 = int(l[1]*512), int(l[2]*512)
											corrds += str(x0) + ' ' + str(y0) + ' '
										out_1 += 1
										out_write.append([person_imgs_id[w[1]], 1, corrds[:-1]])
									else:
										out_write.append([person_imgs_id[w[1]], -1, ''])

							else:
								id_1 = person_imgs_id[w[0]]
								id_2 = person_imgs_id[w[1]]
								id_3 = person_imgs_id[w[2]]
								label_1 = os.path.join(args.exp_path, 'labels', id_1+'.txt')
								label_2 = os.path.join(args.exp_path, 'labels', id_2+'.txt')
								label_3 = os.path.join(args.exp_path, 'labels', id_3+'.txt')

								if not os.path.isfile(label_1) and not os.path.isfile(label_3):
									out_write.append([person_imgs_id[w[1]], -1, ''])
								else:
									if os.path.isfile(label_2):
										f2 = open(label_2,'r')
										ls2 = f2.read().splitlines()
										f2.close()

										corrds = ''
										for l in ls2:
											l = list(map(float, l.split(' ')))
											x0, y0 = int(l[1]*512), int(l[2]*512)
											corrds += str(x0) + ' ' + str(y0) + ' '
										out_1 += 1
										out_write.append([person_imgs_id[w[1]], 1, corrds[:-1]])
									else:
										out_write.append([person_imgs_id[w[1]], -1, ''])

						if out_1 == 0:
							for out in out_write:
								writer.writerow([out[0], -1, ''])
						else:
							for out in out_write:
								writer.writerow(out)


					else:
						for img_id in person_imgs_id:
							if img_id in persons:
								label_path = os.path.join(args.exp_path, 'labels', img_id+'.txt')
								f =  open(label_path,'r')
								ls = f.read().splitlines()

								corrds = ''
								for l in ls:
									l = list(map(float, l.split(' ')))
									x0, y0 = int(l[1]*512), int(l[2]*512)
									corrds += str(x0) + ' ' + str(y0) + ' '

								f.close()

								writer.writerow([img_id, 1, corrds[:-1]])
							else:
								writer.writerow([img_id, -1, ''])

			else:
				### predicts 0
				for _id in person_imgs_id:
					writer.writerow([_id, 0, ''])


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--exp_path', type=str)
	parser.add_argument('--window', action='store_true')
	parser.add_argument('--output_csv', type=str, default='test_out.csv')
	args = parser.parse_args()
	main(args)

	





